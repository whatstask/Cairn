//! The archive of everything tried, and how the engine decides what to try
//! next: which operator, which parents, which proposer. These choices
//! calibrate themselves from which past proposals actually won.

use std::collections::BTreeMap;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::rng::Rng;
use crate::state::{ProgramState, StrategyRecord};
use crate::stats::Counts;
use crate::types::{Operator, Stage, StrategyId};

/// Track record of an operator or proposer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrackRecord {
    pub proposed: u32,
    pub reached_traffic: u32,
    pub promoted: u32,
    pub failed: u32,
}

impl TrackRecord {
    /// Posterior mean win rate with a Beta(1, 1) prior. A strategy that got
    /// traffic and was not rolled back earns half a win, so credit flows
    /// before slow promotions arrive.
    pub fn score(&self) -> f64 {
        let wins = self.promoted as f64 + 0.5 * self.reached_traffic.saturating_sub(self.promoted) as f64;
        let trials =
            (self.promoted + self.failed) as f64 + 0.5 * self.reached_traffic.saturating_sub(self.promoted) as f64;
        (1.0 + wins) / (2.0 + trials)
    }

    fn record(&mut self, r: &StrategyRecord) {
        self.proposed += 1;
        if r.live_at.is_some() {
            self.reached_traffic += 1;
        }
        match &r.stage {
            Stage::Incumbent => self.promoted += 1,
            Stage::Retired { reason } if reason.starts_with("replaced by") => self.promoted += 1,
            Stage::Retired { .. } | Stage::RolledBack { .. } | Stage::Rejected { .. } => self.failed += 1,
            _ => {}
        }
    }
}

pub fn operator_records(state: &ProgramState) -> BTreeMap<Operator, TrackRecord> {
    let mut out: BTreeMap<Operator, TrackRecord> = Operator::ALL.iter().map(|o| (*o, TrackRecord::default())).collect();
    for r in state.strategies.values() {
        if let Some(op) = r.strategy.operator {
            out.entry(op).or_default().record(r);
        }
    }
    out
}

pub fn proposer_records(state: &ProgramState) -> BTreeMap<String, TrackRecord> {
    let mut out: BTreeMap<String, TrackRecord> = BTreeMap::new();
    for r in state.strategies.values() {
        if let Some(p) = &r.strategy.proposer {
            out.entry(p.clone()).or_default().record(r);
        }
    }
    out
}

/// True when the program needs bigger jumps: drift since the last
/// promotion, or two or more challengers lost with no promotion in 60 days.
pub fn stagnating(state: &ProgramState, now: DateTime<Utc>) -> bool {
    if let Some(d) = state.last_drift {
        if state.last_promotion.is_none_or(|p| p < d) {
            return true;
        }
    }
    let since = state.last_promotion.unwrap_or(state.created_at);
    let losses = state
        .strategies
        .values()
        .filter(|r| r.stage_at > since && matches!(r.stage, Stage::Retired { .. } | Stage::RolledBack { .. }))
        .filter(|r| r.strategy.id != state.incumbent)
        .count();
    losses >= 2 && now - since > Duration::days(60)
}

/// Operator probabilities: a base mix (wider when stagnating), tilted by each
/// operator's track record, with a 5% floor so none is ever abandoned.
pub fn operator_mix(state: &ProgramState, now: DateTime<Utc>) -> Vec<(Operator, f64)> {
    let base: [f64; 4] = if stagnating(state, now) { [0.25, 0.35, 0.10, 0.30] } else { [0.50, 0.25, 0.10, 0.15] };
    let records = operator_records(state);
    let mean_score = records.values().map(TrackRecord::score).sum::<f64>() / records.len() as f64;
    let mut weights: Vec<(Operator, f64)> = Operator::ALL
        .iter()
        .zip(base)
        .map(|(op, b)| (*op, b * records.get(op).map(TrackRecord::score).unwrap_or(0.5) / mean_score))
        .collect();
    let total: f64 = weights.iter().map(|w| w.1).sum();
    for w in &mut weights {
        w.1 = (w.1 / total).max(0.05);
    }
    let total: f64 = weights.iter().map(|w| w.1).sum();
    weights.iter().map(|(o, w)| (*o, w / total)).collect()
}

pub fn sample_operator(mix: &[(Operator, f64)], rng: &mut Rng) -> Operator {
    let mut u = rng.uniform();
    for (op, p) in mix {
        if u < *p {
            return *op;
        }
        u -= p;
    }
    mix.last().map(|m| m.0).unwrap_or(Operator::Local)
}

/// All-time matured evidence for a strategy that has had traffic.
pub fn lifetime_counts(state: &ProgramState, id: &str, now: DateTime<Utc>) -> Counts {
    state.matured_counts(id, state.created_at, now, false)
}

/// Parent sampling in the style of the Darwin Gödel Machine: weight by
/// posterior mean, divided by `1 + children`, so strong lines are explored
/// without starving the rest.
pub fn sample_parents(state: &ProgramState, count: usize, now: DateTime<Utc>, rng: &mut Rng) -> Vec<StrategyId> {
    let mut pool: Vec<(StrategyId, f64)> = state
        .strategies
        .values()
        .filter(|r| r.live_at.is_some() && !matches!(r.stage, Stage::RolledBack { .. }))
        .map(|r| {
            let (a, b) = state.prior.posterior(lifetime_counts(state, &r.strategy.id, now));
            (r.strategy.id.clone(), (a / (a + b)) / (1.0 + r.children as f64))
        })
        .collect();
    let mut out = Vec::new();
    while out.len() < count && !pool.is_empty() {
        let total: f64 = pool.iter().map(|p| p.1).sum();
        let mut u = rng.uniform() * total;
        let mut pick = pool.len() - 1;
        for (i, p) in pool.iter().enumerate() {
            if u < p.1 {
                pick = i;
                break;
            }
            u -= p.1;
        }
        out.push(pool.remove(pick).0);
    }
    out
}
