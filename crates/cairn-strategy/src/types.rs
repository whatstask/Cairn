use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::stats::BetaPrior;

pub type StrategyId = String;

/// A playbook an agent can follow, as data.
///
/// Strategies are immutable once proposed. Improvements are new strategies
/// that name their `parent`, so the archive keeps a full lineage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Strategy {
    /// Assigned by the engine when the strategy is proposed.
    #[serde(default)]
    pub id: StrategyId,
    pub name: String,
    /// Where the strategy sits in the archive, e.g.
    /// `{"segment": "fintech CFOs", "channel": "email", "offer": "audit"}`.
    #[serde(default)]
    pub descriptor: BTreeMap<String, String>,
    /// Itemized playbook. Edits are small and itemized rather than wholesale
    /// rewrites, which keeps playbooks from collapsing over many revisions.
    pub playbook: Vec<String>,
    /// Why this should work, stated so a test can refute it.
    #[serde(default)]
    pub hypothesis: Option<String>,
    #[serde(default)]
    pub parents: Vec<StrategyId>,
    #[serde(default)]
    pub operator: Option<Operator>,
    /// Who proposed it: a human, or a proposer such as `model:claude-opus-5-5`.
    #[serde(default)]
    pub proposer: Option<String>,
    /// The proposer's predicted relative lift over the incumbent. Untrusted:
    /// it only orders the queue, after calibration against real outcomes.
    #[serde(default)]
    pub predicted_lift: Option<f64>,
}

impl Strategy {
    pub fn new(name: impl Into<String>, playbook: Vec<String>) -> Self {
        Self {
            id: String::new(),
            name: name.into(),
            descriptor: BTreeMap::new(),
            playbook,
            hypothesis: None,
            parents: Vec::new(),
            operator: None,
            proposer: None,
            predicted_lift: None,
        }
    }

    pub fn with_descriptor(mut self, axis: impl Into<String>, value: impl Into<String>) -> Self {
        self.descriptor.insert(axis.into(), value.into());
        self
    }

    /// The archive cell, e.g. `channel=email|segment=fintech`.
    pub fn cell(&self) -> String {
        self.descriptor.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("|")
    }
}

/// How a strategy was derived from the archive. Wider operators explore
/// further from what is known to work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operator {
    /// Small edits inside the parent's cell (wording, cadence, one bullet).
    Local,
    /// Move one descriptor axis (a new segment, channel, offer or angle).
    AxisShift,
    /// Combine two strong parents.
    Crossover,
    /// A fresh strategy from the goal alone, aimed at an empty cell.
    Fresh,
}

impl Operator {
    pub const ALL: [Operator; 4] = [Operator::Local, Operator::AxisShift, Operator::Crossover, Operator::Fresh];
}

/// Where a strategy is in its life.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum Stage {
    /// Passed screening; waiting for a human decision.
    AwaitingApproval,
    /// Approved; waits for a canary slot.
    Approved,
    /// Live as a challenger on a capped share of traffic.
    Canary,
    /// The strategy most traffic gets.
    Incumbent,
    Rejected {
        reason: String,
    },
    Retired {
        reason: String,
    },
    RolledBack {
        reason: String,
    },
}

impl Stage {
    pub fn is_live(&self) -> bool {
        matches!(self, Stage::Canary | Stage::Incumbent)
    }

    pub fn is_closed(&self) -> bool {
        matches!(self, Stage::Rejected { .. } | Stage::Retired { .. } | Stage::RolledBack { .. })
    }
}

/// A rate that must stay below a limit (spam complaints, unsubscribes,
/// bounces). Breaching it rolls a challenger back and pauses a program whose
/// incumbent breaches it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Guardrail {
    pub event: String,
    pub max_rate: f64,
}

/// Everything a program's decisions depend on. Only humans change it; the
/// engine's self-calibration never touches the metric contract, guardrails
/// or approval rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProgramConfig {
    /// What the program optimizes, in plain words; shown to proposers.
    pub goal: String,
    /// The outcome event that counts as success (e.g. `meeting_booked`).
    pub success_event: String,
    pub guardrails: Vec<Guardrail>,
    /// Share of subjects kept on the frozen baseline to measure real lift and
    /// catch slow harms.
    pub holdout_fraction: f64,
    pub max_challengers: usize,
    /// Challengers may cost at most this share of the incumbent's expected
    /// successes (conservative bandit, Wu et al. 2016).
    pub conservative_alpha: f64,
    /// Forced exploration per live challenger, inside the conservative budget.
    pub exploration_floor: f64,
    /// A challenger never gets more than this share of traffic. It starts at
    /// `canary_initial_share` and ramps up as clean guardrail evidence comes in.
    pub canary_max_share: f64,
    pub canary_initial_share: f64,
    /// Error rate of promotion and rollback decisions (anytime-valid).
    pub decision_alpha: f64,
    /// Matured outcomes each side needs before any promotion decision.
    pub min_samples: f64,
    /// A challenger that has not won after this many matured outcomes, or
    /// this many days, retires so the slot goes to a new idea.
    pub max_samples: f64,
    pub max_canary_days: f64,
    /// Expected days between false drift alarms on the incumbent.
    pub drift_alarm_days: f64,
    /// Starting hours after assignment before guardrail rates are counted;
    /// self-calibrated from how fast guardrail events actually arrive.
    pub guardrail_window_hours: f64,
    /// Starting maturity window (hours); self-calibrated from observed lags.
    pub maturity_hours: f64,
    pub maturity_bounds_hours: (f64, f64),
    /// Starting prior; self-calibrated by empirical Bayes.
    pub prior: BetaPrior,
    pub prior_strength_bounds: (f64, f64),
    /// Half-life (days) for discounting old evidence in allocation.
    pub half_life_days: Option<f64>,
    /// Descriptor axes whose change counts as scope expansion and always
    /// needs a human.
    pub scope_axes: Vec<String>,
    /// Auto-approve `Local` edits that pass screening. Off by default: every
    /// change waits for a human.
    pub auto_approve_local: bool,
    /// Token similarity at or above which a proposal duplicates an archived
    /// strategy (including failed ones).
    pub novelty_threshold: f64,
    /// Calibrated parameters move at most this factor per update.
    pub calibration_max_step: f64,
    /// Thompson draws used to estimate each assignment's propensity.
    pub propensity_samples: usize,
}

impl Default for ProgramConfig {
    fn default() -> Self {
        Self {
            goal: String::new(),
            success_event: "success".into(),
            guardrails: Vec::new(),
            holdout_fraction: 0.05,
            max_challengers: 3,
            conservative_alpha: 0.10,
            exploration_floor: 0.05,
            canary_max_share: 0.25,
            canary_initial_share: 0.10,
            decision_alpha: 0.05,
            min_samples: 50.0,
            max_samples: 3000.0,
            max_canary_days: 90.0,
            drift_alarm_days: 365.0,
            guardrail_window_hours: 72.0,
            maturity_hours: 24.0 * 7.0,
            maturity_bounds_hours: (24.0, 24.0 * 45.0),
            prior: BetaPrior::default(),
            prior_strength_bounds: (2.0, 20.0),
            half_life_days: Some(70.0),
            scope_axes: vec!["segment".into(), "channel".into(), "geo".into()],
            auto_approve_local: false,
            novelty_threshold: 0.9,
            calibration_max_step: 1.5,
            propensity_samples: 256,
        }
    }
}

/// What the engine tells the agent to do for one subject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assignment {
    pub assignment_id: String,
    pub strategy_id: StrategyId,
    pub strategy: Strategy,
    /// Probability this strategy was chosen; logged for off-policy evaluation.
    pub propensity: f64,
    pub holdout: bool,
    /// The program is paused (kill switch). Callers must stop acting.
    pub halt: bool,
}
