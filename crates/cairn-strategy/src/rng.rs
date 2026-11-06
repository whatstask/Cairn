//! A small deterministic PRNG. Every random decision is seeded from its own
//! id, so replaying a program's events reproduces every sample exactly.

use sha2::{Digest, Sha256};

/// SplitMix64. Not cryptographic; fast, tiny and good enough for sampling.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn seeded(seed: u64) -> Self {
        Self(seed)
    }

    /// Seed from a string key (e.g. an assignment id).
    pub fn from_key(key: &str) -> Self {
        let digest = Sha256::digest(key.as_bytes());
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[..8]);
        Self(u64::from_le_bytes(bytes))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform index in 0..n. `n` must be non-zero.
    pub fn index(&mut self, n: usize) -> usize {
        ((self.uniform() * n as f64) as usize).min(n - 1)
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Gamma(shape, 1) by Marsaglia and Tsang.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        if shape <= 0.0 {
            return 0.0;
        }
        if shape < 1.0 {
            let u = self.uniform();
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.uniform();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Beta(a, b).
    pub fn beta(&mut self, a: f64, b: f64) -> f64 {
        let x = self.gamma(a);
        let y = self.gamma(b);
        if x + y == 0.0 { 0.5 } else { x / (x + y) }
    }

    /// Exponential with the given mean.
    pub fn exponential(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.uniform()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_key_same_stream() {
        let mut a = Rng::from_key("asg-1");
        let mut b = Rng::from_key("asg-1");
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn beta_mean_is_right() {
        let mut rng = Rng::seeded(7);
        let n = 20_000;
        let mean: f64 = (0..n).map(|_| rng.beta(3.0, 7.0)).sum::<f64>() / n as f64;
        assert!((mean - 0.3).abs() < 0.01, "mean {mean}");
        let small: f64 = (0..n).map(|_| rng.beta(0.5, 0.5)).sum::<f64>() / n as f64;
        assert!((small - 0.5).abs() < 0.02, "mean {small}");
    }
}
