//! The statistics the decision plane relies on. Everything here is plain,
//! deterministic arithmetic: no model calls, no I/O.

use serde::{Deserialize, Serialize};

const EPS: f64 = 1e-12;

/// Weighted Bernoulli evidence: `successes` out of `trials`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Counts {
    pub trials: f64,
    pub successes: f64,
}

impl Counts {
    pub fn new(trials: f64, successes: f64) -> Self {
        Self { trials, successes }
    }

    pub fn add(&mut self, other: Counts) {
        self.trials += other.trials;
        self.successes += other.successes;
    }

    pub fn rate(&self) -> Option<f64> {
        (self.trials > 0.0).then(|| self.successes / self.trials)
    }
}

/// Beta prior on a conversion rate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BetaPrior {
    pub alpha: f64,
    pub beta: f64,
}

impl Default for BetaPrior {
    fn default() -> Self {
        Self { alpha: 1.0, beta: 1.0 }
    }
}

impl BetaPrior {
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Pseudo-observations the prior is worth.
    pub fn strength(&self) -> f64 {
        self.alpha + self.beta
    }

    pub fn posterior(&self, c: Counts) -> (f64, f64) {
        (self.alpha + c.successes, self.beta + (c.trials - c.successes).max(0.0))
    }
}

/// KL divergence between Bernoulli(p) and Bernoulli(q).
pub fn kl_bernoulli(p: f64, q: f64) -> f64 {
    let p = p.clamp(EPS, 1.0 - EPS);
    let q = q.clamp(EPS, 1.0 - EPS);
    p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln()
}

/// Two-sided, anytime-valid confidence sequence for a mean in [0, 1].
///
/// Uses the normal-mixture boundary for sub-Gaussian increments (variance
/// proxy 1/4) from Howard et al. (2021): the bounds hold simultaneously at
/// every sample size with probability at least `1 - alpha`, so they can be
/// checked after every new outcome without inflating error rates. `n_opt` is
/// the sample size the boundary is tightest around.
pub fn confidence_sequence(c: Counts, alpha: f64, n_opt: f64) -> (f64, f64) {
    if c.trials <= 0.0 {
        return (0.0, 1.0);
    }
    let sigma2 = 0.25;
    let a = (2.0 / alpha).ln();
    let rho = sigma2 * n_opt / (2.0 * a + (1.0 + 2.0 * a).ln());
    let v = sigma2 * c.trials;
    let radius = ((v + rho) * ((v + rho) / rho).ln() + 2.0 * (v + rho) * a).sqrt() / c.trials;
    let mean = c.successes / c.trials;
    ((mean - radius).max(0.0), (mean + radius).min(1.0))
}

/// Natural log of the gamma function (Lanczos, g = 7).
pub fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Anytime-valid confidence sequence for a Bernoulli rate by the
/// prior-posterior ratio (Waudby-Smith and Ramdas, 2020) with a uniform
/// prior: the set of rates where the Beta(1 + s, 1 + f) posterior density
/// exceeds `alpha`. Valid under continuous monitoring, and far tighter than
/// sub-Gaussian bounds when rates are small, which is where conversion and
/// complaint rates live.
pub fn rate_bounds(c: Counts, alpha: f64) -> (f64, f64) {
    if c.trials <= 0.0 {
        return (0.0, 1.0);
    }
    let s = c.successes.max(0.0);
    let f = (c.trials - c.successes).max(0.0);
    let ln_b = ln_gamma(1.0 + s) + ln_gamma(1.0 + f) - ln_gamma(2.0 + s + f);
    let ln_alpha = alpha.ln();
    let inside = |p: f64| {
        let p = p.clamp(1e-15, 1.0 - 1e-15);
        s * p.ln() + f * (1.0 - p).ln() - ln_b > ln_alpha
    };
    let mode = s / (s + f);
    let bisect = |mut out: f64, mut inn: f64| {
        for _ in 0..60 {
            let mid = (out + inn) / 2.0;
            if inside(mid) { inn = mid } else { out = mid }
        }
        inn
    };
    let lower = if inside(0.0) { 0.0 } else { bisect(0.0, mode) };
    let upper = if inside(1.0) { 1.0 } else { bisect(1.0, mode) };
    (lower, upper)
}

/// Anytime-valid one-sided test that a Bernoulli rate exceeds `limit`.
///
/// Mixture of likelihood ratios against rates 1.5 to 10 times the limit.
/// Under any rate at or below the limit it is a nonnegative supermartingale,
/// so by Ville's inequality the chance it ever reaches `1 / alpha` is at most
/// `alpha`, however often it is checked. Returns the e-value.
pub fn exceeds_evalue(c: Counts, limit: f64) -> f64 {
    let limit = limit.clamp(1e-9, 1.0 - 1e-9);
    let failures = (c.trials - c.successes).max(0.0);
    let alts: Vec<f64> =
        [1.5, 2.0, 3.0, 5.0, 10.0].iter().map(|m| (limit * m).min(0.99)).filter(|q| *q > limit).collect();
    if alts.is_empty() {
        return 0.0;
    }
    alts.iter()
        .map(|q| (c.successes * (q / limit).ln() + failures * ((1.0 - q) / (1.0 - limit)).ln()).exp())
        .sum::<f64>()
        / alts.len() as f64
}

/// Fit a Beta prior across arms by the method of moments (empirical Bayes).
///
/// Between-arm variance is the observed variance of rates minus the
/// binomial noise each arm's sample size explains. Returns `None` when fewer
/// than two arms have data. The strength is clamped to `[min, max]`.
pub fn fit_beta_prior(arms: &[Counts], min_strength: f64, max_strength: f64) -> Option<BetaPrior> {
    let arms: Vec<&Counts> = arms.iter().filter(|c| c.trials >= 1.0).collect();
    if arms.len() < 2 {
        return None;
    }
    let total: f64 = arms.iter().map(|c| c.trials).sum();
    let mean = (arms.iter().map(|c| c.successes).sum::<f64>() / total).clamp(1e-4, 1.0 - 1e-4);
    let k = arms.len() as f64;
    let rates: Vec<f64> = arms.iter().map(|c| c.successes / c.trials).collect();
    let rate_mean = rates.iter().sum::<f64>() / k;
    let var = rates.iter().map(|r| (r - rate_mean).powi(2)).sum::<f64>() / (k - 1.0);
    let noise = mean * (1.0 - mean) * arms.iter().map(|c| 1.0 / c.trials).sum::<f64>() / k;
    let between = var - noise;
    let strength = if between <= 0.0 { max_strength } else { mean * (1.0 - mean) / between - 1.0 };
    let strength = strength.clamp(min_strength, max_strength);
    Some(BetaPrior { alpha: mean * strength, beta: (1.0 - mean) * strength })
}

/// A detected change in a sequence of binomial buckets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Change {
    /// Index of the first bucket after the change.
    pub split: usize,
    pub before: f64,
    pub after: f64,
    pub statistic: f64,
    pub threshold: f64,
}

/// Bernoulli generalized likelihood ratio test for one change point
/// (Besson, Kaufmann, Maillard and Seznec, JMLR 2022), run over buckets.
///
/// Fires when `max_k [n1 kl(p1, p) + n2 kl(p2, p)]` exceeds
/// `ln(n^{3/2} / delta)`, the threshold from the paper. Used to scan history,
/// where no reference rate is known in advance.
pub fn glr_change(buckets: &[Counts], delta: f64, min_side: f64) -> Option<Change> {
    let total = buckets.iter().fold(Counts::default(), |mut a, b| {
        a.add(*b);
        a
    });
    if total.trials < 2.0 * min_side {
        return None;
    }
    let p = total.successes / total.trials;
    let threshold = (total.trials.powf(1.5) / delta).ln();
    let mut left = Counts::default();
    let mut best: Option<Change> = None;
    for (i, b) in buckets.iter().enumerate().take(buckets.len().saturating_sub(1)) {
        left.add(*b);
        let right = Counts::new(total.trials - left.trials, total.successes - left.successes);
        if left.trials < min_side || right.trials < min_side {
            continue;
        }
        let p1 = left.successes / left.trials;
        let p2 = right.successes / right.trials;
        let stat = left.trials * kl_bernoulli(p1, p) + right.trials * kl_bernoulli(p2, p);
        if best.as_ref().is_none_or(|c| stat > c.statistic) {
            best = Some(Change { split: i + 1, before: p1, after: p2, statistic: stat, threshold });
        }
    }
    best.filter(|c| c.statistic > c.threshold)
}

/// Mixture e-CUSUM detector for a *drop* in a Bernoulli rate below a
/// reference (Shin, Ramdas and Rinaldo, "E-detectors", 2023).
///
/// Each alternative `p1 = reference * (1 - d)` runs its own CUSUM-style
/// e-process `C = max(C, 1) * LR`; their average is again an e-detector, so
/// alarming when it reaches `1 / alpha` keeps the expected time to a false
/// alarm at least `1 / alpha` buckets while the rate stays at or above the
/// reference. The threshold comes from the false-alarm budget alone; nothing
/// is tuned on data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DropDetector {
    pub reference: f64,
    pub drops: Vec<f64>,
    pub state: Vec<f64>,
}

impl DropDetector {
    /// Watches for relative drops of 20%, 35% and 50% (a mixture, so any drop
    /// in that range is caught quickly).
    pub fn new(reference: f64) -> Self {
        let drops = vec![0.2, 0.35, 0.5];
        let state = vec![0.0; drops.len()];
        Self { reference: reference.clamp(1e-6, 1.0 - 1e-6), drops, state }
    }

    /// Feed one bucket; returns the current detector value.
    pub fn update(&mut self, bucket: Counts) -> f64 {
        let p0 = self.reference;
        let failures = (bucket.trials - bucket.successes).max(0.0);
        for (c, d) in self.state.iter_mut().zip(&self.drops) {
            let p1 = p0 * (1.0 - d);
            let log_lr = bucket.successes * (p1 / p0).ln() + failures * ((1.0 - p1) / (1.0 - p0)).ln();
            *c = c.max(1.0) * log_lr.exp();
        }
        self.value()
    }

    pub fn value(&self) -> f64 {
        self.state.iter().sum::<f64>() / self.state.len() as f64
    }
}

/// Standard normal CDF (Abramowitz and Stegun 7.1.26, error < 1.5e-7).
pub fn normal_cdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * z.abs() / std::f64::consts::SQRT_2);
    let poly =
        t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-(z * z) / 2.0).exp();
    if z >= 0.0 { 0.5 * (1.0 + erf) } else { 0.5 * (1.0 - erf) }
}

/// Posterior probability that arm `a` has the higher rate, with Beta(1, 1)
/// priors and a normal approximation to the difference.
pub fn prob_better(a: Counts, b: Counts) -> f64 {
    let moments = |c: Counts| {
        let (x, y) = (1.0 + c.successes, 1.0 + (c.trials - c.successes).max(0.0));
        let n = x + y;
        (x / n, x * y / (n * n * (n + 1.0)))
    };
    let ((ma, va), (mb, vb)) = (moments(a), moments(b));
    normal_cdf((ma - mb) / (va + vb).sqrt().max(1e-12))
}

/// Spearman rank correlation; `None` with fewer than three pairs.
pub fn spearman(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < 3 {
        return None;
    }
    let rank = |vals: Vec<f64>| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..vals.len()).collect();
        idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        let mut ranks = vec![0.0; vals.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && vals[idx[j + 1]] == vals[idx[i]] {
                j += 1;
            }
            let r = (i + j) as f64 / 2.0;
            for k in &idx[i..=j] {
                ranks[*k] = r;
            }
            i = j + 1;
        }
        ranks
    };
    let a = rank(pairs.iter().map(|p| p.0).collect());
    let b = rank(pairs.iter().map(|p| p.1).collect());
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    if va == 0.0 || vb == 0.0 { None } else { Some(cov / (va * vb).sqrt()) }
}

/// Isotonic (monotone non-decreasing) regression by pool-adjacent-violators.
/// Returns breakpoints `(x, fitted y)` sorted by `x`, used to map an
/// untrusted predictor's scores onto realized outcomes.
pub fn isotonic(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Blocks of (sum_y, count, max_x).
    let mut blocks: Vec<(f64, f64, f64)> = Vec::new();
    for (x, y) in pts {
        blocks.push((y, 1.0, x));
        while blocks.len() > 1 {
            let n = blocks.len();
            let (s1, c1, _) = blocks[n - 2];
            let (s2, c2, x2) = blocks[n - 1];
            if s1 / c1 <= s2 / c2 {
                break;
            }
            blocks.truncate(n - 2);
            blocks.push((s1 + s2, c1 + c2, x2));
        }
    }
    blocks.into_iter().map(|(s, c, x)| (x, s / c)).collect()
}

/// Evaluate an isotonic fit at `x` (step function, clamped at the ends).
pub fn isotonic_eval(fit: &[(f64, f64)], x: f64) -> Option<f64> {
    fit.iter().find(|(bx, _)| x <= *bx).or(fit.last()).map(|(_, y)| *y)
}

/// Outcomes per arm needed to detect an absolute lift `delta` on base rate
/// `p` with ~80% power at 5% two-sided (the standard 16 p(1-p) / delta^2).
pub fn samples_for_lift(p: f64, delta: f64) -> f64 {
    16.0 * p * (1.0 - p) / (delta * delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn confidence_sequence_covers_under_continuous_monitoring() {
        let alpha = 0.1;
        let p = 0.12;
        let mut misses = 0;
        let runs = 400;
        for seed in 0..runs {
            let mut rng = Rng::seeded(seed);
            let mut c = Counts::default();
            let mut missed = false;
            for _ in 0..2000 {
                c.add(Counts::new(1.0, if rng.uniform() < p { 1.0 } else { 0.0 }));
                let (lo, hi) = confidence_sequence(c, alpha, 500.0);
                if p < lo || p > hi {
                    missed = true;
                    break;
                }
            }
            misses += missed as u32;
        }
        assert!((misses as f64 / runs as f64) <= alpha, "missed {misses}/{runs}");
    }

    #[test]
    fn rate_bounds_cover_small_rates_and_are_tight() {
        let alpha = 0.1;
        let p = 0.01;
        let mut misses = 0;
        let runs = 300;
        for seed in 0..runs {
            let mut rng = Rng::seeded(77 + seed);
            let mut c = Counts::default();
            for i in 0..3000 {
                c.add(Counts::new(1.0, if rng.uniform() < p { 1.0 } else { 0.0 }));
                if i % 10 == 0 {
                    let (lo, hi) = rate_bounds(c, alpha);
                    if p < lo || p > hi {
                        misses += 1;
                        break;
                    }
                }
            }
        }
        assert!((misses as f64 / runs as f64) <= alpha, "missed {misses}/{runs}");
        // At 3% over 400 trials, a 0.5% limit is clearly exceeded.
        let (lo, _) = rate_bounds(Counts::new(400.0, 12.0), alpha);
        assert!(lo > 0.005, "lower {lo}");
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn glr_finds_a_step_and_stays_quiet_without_one() {
        let mut rng = Rng::seeded(3);
        let day = |rng: &mut Rng, p: f64| {
            let s = (0..40).filter(|_| rng.uniform() < p).count() as f64;
            Counts::new(40.0, s)
        };
        let mut buckets: Vec<Counts> = (0..30).map(|_| day(&mut rng, 0.2)).collect();
        buckets.extend((0..30).map(|_| day(&mut rng, 0.08)));
        let change = glr_change(&buckets, 0.02, 20.0).expect("change");
        assert!((25..=35).contains(&change.split), "split {}", change.split);

        let mut alarms = 0;
        for seed in 0..100 {
            let mut rng = Rng::seeded(1000 + seed);
            let quiet: Vec<Counts> = (0..60).map(|_| day(&mut rng, 0.2)).collect();
            alarms += glr_change(&quiet, 0.02, 20.0).is_some() as u32;
        }
        assert!(alarms <= 5, "false alarms {alarms}/100");
    }

    #[test]
    fn drop_detector_alarms_on_drop_but_rarely_without() {
        let threshold = 100.0;
        let mut rng = Rng::seeded(9);
        let mut d = DropDetector::new(0.2);
        let mut fired_at = None;
        for day in 0..120 {
            let p = if day < 60 { 0.2 } else { 0.1 };
            let s = (0..30).filter(|_| rng.uniform() < p).count() as f64;
            if d.update(Counts::new(30.0, s)) >= threshold {
                fired_at = Some(day);
                break;
            }
        }
        let day = fired_at.expect("drop detected");
        assert!((60..90).contains(&day), "fired on day {day}");

        let mut false_alarms = 0;
        for seed in 0..50 {
            let mut rng = Rng::seeded(500 + seed);
            let mut d = DropDetector::new(0.2);
            for _ in 0..100 {
                let s = (0..30).filter(|_| rng.uniform() < 0.2).count() as f64;
                if d.update(Counts::new(30.0, s)) >= threshold {
                    false_alarms += 1;
                    break;
                }
            }
        }
        // ARL >= 100 buckets, so over 100 buckets at most ~63% could fire in
        // the worst case; in practice it is far rarer.
        assert!(false_alarms <= 10, "false alarms {false_alarms}/50");
    }

    #[test]
    fn empirical_bayes_shrinks_strength_with_real_spread() {
        let tight = [Counts::new(1000.0, 100.0), Counts::new(1000.0, 102.0), Counts::new(1000.0, 98.0)];
        let wide = [Counts::new(1000.0, 50.0), Counts::new(1000.0, 150.0), Counts::new(1000.0, 250.0)];
        let t = fit_beta_prior(&tight, 2.0, 50.0).unwrap();
        let w = fit_beta_prior(&wide, 2.0, 50.0).unwrap();
        assert!(t.strength() > w.strength());
        assert!((w.mean() - 0.15).abs() < 1e-9);
    }

    #[test]
    fn exceeds_evalue_is_valid_and_fast() {
        // Under the null (rate at the limit) it rarely reaches 1 / alpha.
        let (limit, alpha) = (0.005, 0.05);
        let mut rejections = 0;
        for seed in 0..300 {
            let mut rng = Rng::seeded(900 + seed);
            let mut c = Counts::default();
            for _ in 0..3000 {
                c.add(Counts::new(1.0, (rng.uniform() < limit) as u8 as f64));
                if exceeds_evalue(c, limit) >= 1.0 / alpha {
                    rejections += 1;
                    break;
                }
            }
        }
        assert!(rejections as f64 / 300.0 <= alpha, "rejected {rejections}/300");
        // At six times the limit it fires within a couple of hundred trials.
        assert!(exceeds_evalue(Counts::new(200.0, 6.0), limit) >= 1.0 / alpha);
    }

    #[test]
    fn prob_better_is_sensible() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!(prob_better(Counts::new(500.0, 100.0), Counts::new(500.0, 50.0)) > 0.99);
        assert!(prob_better(Counts::new(200.0, 6.0), Counts::new(2000.0, 160.0)) < 0.05);
    }

    #[test]
    fn isotonic_is_monotone() {
        let fit = isotonic(&[(0.1, 0.3), (0.2, 0.1), (0.3, 0.2), (0.5, 0.6)]);
        assert!(fit.windows(2).all(|w| w[0].1 <= w[1].1));
        assert_eq!(isotonic_eval(&fit, 0.5), Some(0.6));
    }
}
