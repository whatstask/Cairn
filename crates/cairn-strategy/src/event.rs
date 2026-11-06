use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::types::{ProgramConfig, Strategy, StrategyId};

/// One immutable fact in a program's history. Hash-chained: each event's
/// `hash` covers its content and the previous event's hash, so any edit to
/// the log is detectable by [`verify_chain`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyEvent {
    pub program: String,
    pub seq: u64,
    pub at: DateTime<Utc>,
    pub kind: StrategyEventKind,
    pub hash: String,
}

impl StrategyEvent {
    pub fn new(program: &str, seq: u64, at: DateTime<Utc>, kind: StrategyEventKind, prev_hash: &str) -> Self {
        let hash = chain_hash(program, seq, at, &kind, prev_hash);
        Self { program: program.to_owned(), seq, at, kind, hash }
    }
}

fn chain_hash(program: &str, seq: u64, at: DateTime<Utc>, kind: &StrategyEventKind, prev: &str) -> String {
    let body = serde_json::to_string(kind).unwrap_or_default();
    let mut h = Sha256::new();
    for part in [prev, program, &seq.to_string(), &at.to_rfc3339(), &body] {
        h.update(part.as_bytes());
        h.update([0]);
    }
    hex::encode(h.finalize())
}

/// Check the hash chain; returns the first broken `seq`, if any.
pub fn verify_chain(events: &[StrategyEvent]) -> Result<(), u64> {
    let mut prev = String::new();
    for (i, e) in events.iter().enumerate() {
        if e.seq != i as u64 || chain_hash(&e.program, e.seq, e.at, &e.kind, &prev) != e.hash {
            return Err(e.seq);
        }
        prev = e.hash.clone();
    }
    Ok(())
}

/// A shift in one outcome's rate around a detected change; the
/// deterministic part of "why did it drop".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shift {
    pub event: String,
    pub before: f64,
    pub after: f64,
    /// Two-proportion z-score of the change.
    pub z: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StrategyEventKind {
    ProgramCreated {
        config: ProgramConfig,
        baseline: Strategy,
    },
    ConfigChanged {
        config: ProgramConfig,
        by: String,
    },
    StrategyProposed {
        strategy: Strategy,
    },
    StrategyScreened {
        id: StrategyId,
        passed: bool,
        reasons: Vec<String>,
        needs_approval: bool,
    },
    StrategyApproved {
        id: StrategyId,
        by: String,
        note: Option<String>,
    },
    StrategyRejected {
        id: StrategyId,
        by: Option<String>,
        reason: String,
    },
    CanaryStarted {
        id: StrategyId,
    },
    Assigned {
        assignment_id: String,
        subject: String,
        strategy: StrategyId,
        propensity: f64,
        holdout: bool,
        /// Pessimistic expected cost of this choice versus the incumbent;
        /// summed to enforce the conservative budget.
        #[serde(default)]
        loss: f64,
        /// Budget this play earns: `conservative_alpha` times the incumbent's
        /// expected rate at the time. Challengers may only spend what has
        /// been earned.
        #[serde(default)]
        earned: f64,
        /// `seq` of the calibration in force, so every decision can be tied
        /// to the exact parameters that made it.
        #[serde(default)]
        calibration_seq: u64,
    },
    OutcomeObserved {
        assignment_id: String,
        event: String,
        occurred_at: DateTime<Utc>,
    },
    /// The incumbent's reference rate for drift monitoring, set once enough
    /// outcomes have matured since `from`.
    DriftReferenceSet {
        strategy: StrategyId,
        reference: f64,
        from: DateTime<Utc>,
    },
    DriftDetected {
        strategy: StrategyId,
        reference: f64,
        recent: f64,
        detector: f64,
        diagnosis: Vec<Shift>,
    },
    GuardrailBreached {
        strategy: StrategyId,
        event: String,
        lower_bound: f64,
        limit: f64,
    },
    StrategyPromoted {
        id: StrategyId,
        replaced: StrategyId,
        challenger_rate: f64,
        incumbent_rate: f64,
    },
    StrategyRetired {
        id: StrategyId,
        reason: String,
    },
    StrategyRolledBack {
        id: StrategyId,
        reason: String,
    },
    /// A self-calibrated parameter changed. Bounded and journaled so every
    /// change is auditable and reversible.
    Calibrated {
        param: String,
        old: Value,
        new: Value,
        evidence: String,
    },
    ProposalsRequested {
        reason: String,
    },
    Paused {
        by: String,
        reason: String,
    },
    Resumed {
        by: String,
    },
}

impl StrategyEventKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ProgramCreated { .. } => "program_created",
            Self::ConfigChanged { .. } => "config_changed",
            Self::StrategyProposed { .. } => "strategy_proposed",
            Self::StrategyScreened { .. } => "strategy_screened",
            Self::StrategyApproved { .. } => "strategy_approved",
            Self::StrategyRejected { .. } => "strategy_rejected",
            Self::CanaryStarted { .. } => "canary_started",
            Self::Assigned { .. } => "assigned",
            Self::OutcomeObserved { .. } => "outcome_observed",
            Self::DriftReferenceSet { .. } => "drift_reference_set",
            Self::DriftDetected { .. } => "drift_detected",
            Self::GuardrailBreached { .. } => "guardrail_breached",
            Self::StrategyPromoted { .. } => "strategy_promoted",
            Self::StrategyRetired { .. } => "strategy_retired",
            Self::StrategyRolledBack { .. } => "strategy_rolled_back",
            Self::Calibrated { .. } => "calibrated",
            Self::ProposalsRequested { .. } => "proposals_requested",
            Self::Paused { .. } => "paused",
            Self::Resumed { .. } => "resumed",
        }
    }
}
