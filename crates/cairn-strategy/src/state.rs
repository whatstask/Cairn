use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::event::{StrategyEvent, StrategyEventKind};
use crate::stats::{BetaPrior, Counts};
use crate::types::{ProgramConfig, Stage, Strategy, StrategyId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyRecord {
    pub strategy: Strategy,
    pub stage: Stage,
    pub proposed_at: DateTime<Utc>,
    pub stage_at: DateTime<Utc>,
    /// When it went live (canary start, or creation for the baseline).
    pub live_at: Option<DateTime<Utc>>,
    /// Proposals that named this strategy as a parent.
    pub children: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssignmentRecord {
    pub subject: String,
    pub strategy: StrategyId,
    pub at: DateTime<Utc>,
    pub propensity: f64,
    pub holdout: bool,
    /// Outcome events in arrival order (each name at most once).
    pub events: Vec<(String, DateTime<Utc>)>,
}

impl AssignmentRecord {
    pub fn event_at(&self, name: &str) -> Option<DateTime<Utc>> {
        self.events.iter().find(|(e, _)| e == name).map(|(_, t)| *t)
    }

    pub fn age_hours(&self, now: DateTime<Utc>) -> f64 {
        (now - self.at).num_seconds() as f64 / 3600.0
    }

    /// Outcome once the full window has passed: whether `event` happened
    /// within it. `None` while the window is still open, even if the event
    /// already happened; counting early successes but not yet-unknown
    /// failures would bias decisions toward arms with recent traffic.
    pub fn matured(&self, event: &str, window: f64, now: DateTime<Utc>) -> Option<bool> {
        if self.age_hours(now) < window {
            return None;
        }
        Some(self.event_at(event).is_some_and(|t| (t - self.at).num_seconds() as f64 / 3600.0 <= window))
    }

    /// `Some(true)` if `event` happened within `window` hours, `Some(false)`
    /// once the window has passed without it, `None` while pending. For
    /// allocation only; decisions use [`matured`](Self::matured).
    pub fn within(&self, event: &str, window: f64, now: DateTime<Utc>) -> Option<bool> {
        if let Some(t) = self.event_at(event) {
            if (t - self.at).num_seconds() as f64 / 3600.0 <= window {
                return Some(true);
            }
        }
        (self.age_hours(now) >= window).then_some(false)
    }
}

/// Drift monitoring for the incumbent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriftMonitor {
    pub strategy: StrategyId,
    /// Unset until enough outcomes have matured since `from`.
    pub reference: Option<f64>,
    pub from: DateTime<Utc>,
}

/// The full state of a strategy program, derived only from its events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramState {
    pub name: String,
    pub config: ProgramConfig,
    pub next_seq: u64,
    pub last_hash: String,
    pub created_at: DateTime<Utc>,
    pub strategies: BTreeMap<StrategyId, StrategyRecord>,
    pub incumbent: StrategyId,
    /// The frozen strategy the holdout keeps getting.
    pub baseline: StrategyId,
    pub assignments: BTreeMap<String, AssignmentRecord>,
    /// Latest assignment per subject, so a subject stays on one strategy.
    pub subjects: BTreeMap<String, String>,
    /// Non-holdout assignments so far.
    pub plays: u64,
    /// Sum of the pessimistic losses challengers have been charged.
    pub loss_spent: f64,
    /// Sum of the budget plays have earned.
    pub budget_earned: f64,
    /// Self-calibrated parameters.
    pub maturity_hours: f64,
    pub guardrail_window_hours: f64,
    pub prior: BetaPrior,
    /// Isotonic map from predicted lift to realized lift.
    pub prediction_fit: Vec<(f64, f64)>,
    /// How far predictions are trusted when ordering the queue (0 to 1).
    pub prediction_trust: f64,
    pub calibration_seq: u64,
    pub last_calibrated: BTreeMap<String, DateTime<Utc>>,
    pub monitor: DriftMonitor,
    pub paused: bool,
    pub last_proposal_request: Option<DateTime<Utc>>,
    pub last_promotion: Option<DateTime<Utc>>,
    pub last_drift: Option<DateTime<Utc>>,
    /// Outcome shifts found at the last drift alarm; shown to proposers.
    pub last_diagnosis: Vec<crate::event::Shift>,
}

impl ProgramState {
    pub fn from_events(events: &[StrategyEvent]) -> Option<Self> {
        let first = events.first()?;
        let StrategyEventKind::ProgramCreated { config, baseline } = &first.kind else {
            return None;
        };
        let mut s = ProgramState {
            name: first.program.clone(),
            config: config.clone(),
            next_seq: 0,
            last_hash: String::new(),
            created_at: first.at,
            strategies: BTreeMap::new(),
            incumbent: baseline.id.clone(),
            baseline: baseline.id.clone(),
            assignments: BTreeMap::new(),
            subjects: BTreeMap::new(),
            plays: 0,
            loss_spent: 0.0,
            budget_earned: 0.0,
            maturity_hours: config.maturity_hours,
            guardrail_window_hours: config.guardrail_window_hours,
            prior: config.prior,
            prediction_fit: Vec::new(),
            prediction_trust: 0.0,
            calibration_seq: 0,
            last_calibrated: BTreeMap::new(),
            monitor: DriftMonitor { strategy: baseline.id.clone(), reference: None, from: first.at },
            paused: false,
            last_proposal_request: None,
            last_promotion: None,
            last_drift: None,
            last_diagnosis: Vec::new(),
        };
        for e in events {
            s.apply(e);
        }
        Some(s)
    }

    pub fn apply(&mut self, e: &StrategyEvent) {
        self.next_seq = e.seq + 1;
        self.last_hash = e.hash.clone();
        let at = e.at;
        match &e.kind {
            StrategyEventKind::ProgramCreated { baseline, .. } => {
                self.strategies.insert(
                    baseline.id.clone(),
                    StrategyRecord {
                        strategy: baseline.clone(),
                        stage: Stage::Incumbent,
                        proposed_at: at,
                        stage_at: at,
                        live_at: Some(at),
                        children: 0,
                    },
                );
            }
            StrategyEventKind::ConfigChanged { config, .. } => self.config = config.clone(),
            StrategyEventKind::StrategyProposed { strategy } => {
                for p in &strategy.parents {
                    if let Some(r) = self.strategies.get_mut(p) {
                        r.children += 1;
                    }
                }
                self.strategies.insert(
                    strategy.id.clone(),
                    StrategyRecord {
                        strategy: strategy.clone(),
                        stage: Stage::AwaitingApproval,
                        proposed_at: at,
                        stage_at: at,
                        live_at: None,
                        children: 0,
                    },
                );
            }
            StrategyEventKind::StrategyScreened { id, passed, reasons, needs_approval } => {
                let stage = match (passed, needs_approval) {
                    (false, _) => Stage::Rejected { reason: reasons.join("; ") },
                    (true, true) => Stage::AwaitingApproval,
                    (true, false) => Stage::Approved,
                };
                self.set_stage(id, stage, at);
            }
            StrategyEventKind::StrategyApproved { id, .. } => self.set_stage(id, Stage::Approved, at),
            StrategyEventKind::StrategyRejected { id, reason, .. } => {
                self.set_stage(id, Stage::Rejected { reason: reason.clone() }, at)
            }
            StrategyEventKind::CanaryStarted { id } => {
                self.set_stage(id, Stage::Canary, at);
                if let Some(r) = self.strategies.get_mut(id) {
                    r.live_at = Some(at);
                }
            }
            StrategyEventKind::Assigned {
                assignment_id,
                subject,
                strategy,
                propensity,
                holdout,
                loss,
                earned,
                calibration_seq: _,
            } => {
                self.assignments.insert(
                    assignment_id.clone(),
                    AssignmentRecord {
                        subject: subject.clone(),
                        strategy: strategy.clone(),
                        at,
                        propensity: *propensity,
                        holdout: *holdout,
                        events: Vec::new(),
                    },
                );
                self.subjects.insert(subject.clone(), assignment_id.clone());
                if !holdout {
                    self.plays += 1;
                }
                self.loss_spent += loss;
                self.budget_earned += earned;
            }
            StrategyEventKind::OutcomeObserved { assignment_id, event, occurred_at } => {
                if let Some(a) = self.assignments.get_mut(assignment_id) {
                    if a.event_at(event).is_none() {
                        a.events.push((event.clone(), *occurred_at));
                    }
                }
            }
            StrategyEventKind::DriftReferenceSet { strategy, reference, from } => {
                self.monitor = DriftMonitor { strategy: strategy.clone(), reference: Some(*reference), from: *from };
            }
            StrategyEventKind::DriftDetected { strategy, diagnosis, .. } => {
                self.monitor = DriftMonitor { strategy: strategy.clone(), reference: None, from: at };
                self.last_drift = Some(at);
                self.last_diagnosis = diagnosis.clone();
            }
            StrategyEventKind::GuardrailBreached { .. } => {}
            StrategyEventKind::StrategyPromoted { id, replaced, .. } => {
                self.set_stage(replaced, Stage::Retired { reason: format!("replaced by {id}") }, at);
                self.set_stage(id, Stage::Incumbent, at);
                self.incumbent = id.clone();
                self.monitor = DriftMonitor { strategy: id.clone(), reference: None, from: at };
                self.last_promotion = Some(at);
            }
            StrategyEventKind::StrategyRetired { id, reason } => {
                self.set_stage(id, Stage::Retired { reason: reason.clone() }, at)
            }
            StrategyEventKind::StrategyRolledBack { id, reason } => {
                self.set_stage(id, Stage::RolledBack { reason: reason.clone() }, at)
            }
            StrategyEventKind::Calibrated { param, new, .. } => {
                match param.as_str() {
                    "maturity_hours" => {
                        if let Some(v) = new.as_f64() {
                            self.maturity_hours = v;
                            // Rates measured under the old window are not
                            // comparable: re-derive the drift reference from
                            // the same data under the new window.
                            self.monitor.reference = None;
                        }
                    }
                    "guardrail_window_hours" => {
                        if let Some(v) = new.as_f64() {
                            self.guardrail_window_hours = v;
                        }
                    }
                    "prior" => {
                        if let Ok(p) = serde_json::from_value(new.clone()) {
                            self.prior = p;
                        }
                    }
                    "prediction_fit" => {
                        if let Ok(f) = serde_json::from_value(new.clone()) {
                            self.prediction_fit = f;
                        }
                    }
                    "prediction_trust" => {
                        if let Some(v) = new.as_f64() {
                            self.prediction_trust = v;
                        }
                    }
                    _ => {}
                }
                self.calibration_seq = e.seq;
                self.last_calibrated.insert(param.clone(), at);
            }
            StrategyEventKind::ProposalsRequested { .. } => self.last_proposal_request = Some(at),
            StrategyEventKind::Paused { .. } => self.paused = true,
            StrategyEventKind::Resumed { .. } => self.paused = false,
        }
    }

    fn set_stage(&mut self, id: &str, stage: Stage, at: DateTime<Utc>) {
        if let Some(r) = self.strategies.get_mut(id) {
            r.stage = stage;
            r.stage_at = at;
        }
    }

    pub fn stage(&self, id: &str) -> Option<&Stage> {
        self.strategies.get(id).map(|r| &r.stage)
    }

    pub fn challengers(&self) -> Vec<StrategyId> {
        self.strategies.iter().filter(|(_, r)| r.stage == Stage::Canary).map(|(id, _)| id.clone()).collect()
    }

    /// Strategies waiting to run, best calibrated prediction first, then
    /// oldest first.
    pub fn approved_queue(&self) -> Vec<StrategyId> {
        let mut q: Vec<&StrategyRecord> = self.strategies.values().filter(|r| r.stage == Stage::Approved).collect();
        q.sort_by(|a, b| {
            self.calibrated_score(&b.strategy)
                .total_cmp(&self.calibrated_score(&a.strategy))
                .then(a.proposed_at.cmp(&b.proposed_at))
        });
        q.into_iter().map(|r| r.strategy.id.clone()).collect()
    }

    /// A proposal's predicted lift mapped through the isotonic calibration
    /// and scaled by how much predictions have earned trust. Zero until they
    /// have a track record.
    pub fn calibrated_score(&self, s: &Strategy) -> f64 {
        let Some(p) = s.predicted_lift else { return 0.0 };
        let mapped = crate::stats::isotonic_eval(&self.prediction_fit, p).unwrap_or(0.0);
        mapped * self.prediction_trust
    }

    /// Every outcome event name seen so far.
    pub fn event_names(&self) -> BTreeSet<String> {
        self.assignments.values().flat_map(|a| a.events.iter().map(|(e, _)| e.clone())).collect()
    }

    /// Fraction of a success's eventual lag that has typically passed after
    /// `age` hours: the empirical lag CDF, or linear in the maturity window
    /// until there are enough conversions to estimate it.
    pub fn lag_cdf(&self, age: f64) -> f64 {
        let mut lags = self.success_lags();
        lags.sort_by(f64::total_cmp);
        lag_cdf_sorted(&lags, age, self.maturity_hours)
    }

    /// Hours from assignment to success, for every success observed.
    pub fn success_lags(&self) -> Vec<f64> {
        self.success_lags_older_than(None)
    }

    /// Lags from assignments at least `min_age` hours old: a cohort whose
    /// conversions have had time to arrive, so its lags are not censored.
    pub fn success_lags_older_than(&self, min_age: Option<(f64, DateTime<Utc>)>) -> Vec<f64> {
        let ev = &self.config.success_event;
        self.assignments
            .values()
            .filter(|a| min_age.is_none_or(|(hours, now)| a.age_hours(now) >= hours))
            .filter_map(|a| a.event_at(ev).map(|t| (t - a.at).num_seconds() as f64 / 3600.0))
            .filter(|l| *l >= 0.0)
            .collect()
    }

    /// Evidence for allocation: discounted by age, with pending assignments
    /// counted fractionally by how much of their conversion lag has passed.
    pub fn allocation_counts(&self, arm: &str, now: DateTime<Utc>) -> Counts {
        let ev = &self.config.success_event;
        let mut lags = self.success_lags();
        lags.sort_by(f64::total_cmp);
        let mut c = Counts::default();
        for a in self.assignments.values().filter(|a| a.strategy == arm && !a.holdout) {
            let weight = match self.config.half_life_days {
                Some(h) => 0.5f64.powf(a.age_hours(now) / 24.0 / h),
                None => 1.0,
            };
            match a.within(ev, self.maturity_hours, now) {
                Some(true) => c.add(Counts::new(weight, weight)),
                Some(false) => c.add(Counts::new(weight, 0.0)),
                None => c.add(Counts::new(weight * lag_cdf_sorted(&lags, a.age_hours(now), self.maturity_hours), 0.0)),
            }
        }
        c
    }

    /// Matured, undiscounted evidence for decisions, from assignments made at
    /// or after `since`.
    pub fn matured_counts(&self, arm: &str, since: DateTime<Utc>, now: DateTime<Utc>, holdout: bool) -> Counts {
        let ev = &self.config.success_event;
        let mut c = Counts::default();
        for a in self.assignments.values().filter(|a| a.strategy == arm && a.holdout == holdout && a.at >= since) {
            match a.matured(ev, self.maturity_hours, now) {
                Some(true) => c.add(Counts::new(1.0, 1.0)),
                Some(false) => c.add(Counts::new(1.0, 0.0)),
                None => {}
            }
        }
        c
    }

    /// Matured daily buckets (by assignment day) since `since`; the trailing
    /// days that still have pending assignments are left out.
    pub fn matured_buckets(&self, arm: &str, since: DateTime<Utc>, now: DateTime<Utc>) -> Vec<Counts> {
        let ev = &self.config.success_event;
        let mut days: BTreeMap<i64, (Counts, bool)> = BTreeMap::new();
        for a in self.assignments.values().filter(|a| a.strategy == arm && !a.holdout && a.at >= since) {
            let day = (a.at - since).num_days();
            let entry = days.entry(day).or_insert((Counts::default(), true));
            match a.matured(ev, self.maturity_hours, now) {
                Some(s) => entry.0.add(Counts::new(1.0, s as u8 as f64)),
                None => entry.1 = false,
            }
        }
        days.into_values().take_while(|(_, complete)| *complete).map(|(c, _)| c).collect()
    }

    /// Guardrail evidence: assignments older than the guardrail window, and
    /// how many of them produced `event`.
    pub fn guardrail_counts(&self, arm: &str, event: &str, since: DateTime<Utc>, now: DateTime<Utc>) -> Counts {
        let window = self.guardrail_window_hours;
        let mut c = Counts::default();
        for a in self.assignments.values().filter(|a| a.strategy == arm && !a.holdout && a.at >= since) {
            if a.age_hours(now) >= window {
                c.add(Counts::new(1.0, a.event_at(event).is_some() as u8 as f64));
            }
        }
        c
    }

    /// Hours from assignment to each guardrail event, from assignments at
    /// least `min_age` hours old.
    pub fn guardrail_lags(&self, min_age: f64, now: DateTime<Utc>) -> Vec<f64> {
        let names: Vec<&str> = self.config.guardrails.iter().map(|g| g.event.as_str()).collect();
        self.assignments
            .values()
            .filter(|a| a.age_hours(now) >= min_age)
            .flat_map(|a| {
                a.events
                    .iter()
                    .filter(|(e, _)| names.contains(&e.as_str()))
                    .map(move |(_, t)| (*t - a.at).num_seconds() as f64 / 3600.0)
            })
            .collect()
    }

    /// Share of non-holdout assignments since `since` that went to `arm`.
    pub fn share(&self, arm: &str, since: DateTime<Utc>) -> f64 {
        let (mut n, mut k) = (0u64, 0u64);
        for a in self.assignments.values().filter(|a| !a.holdout && a.at >= since) {
            n += 1;
            k += (a.strategy == arm) as u64;
        }
        if n == 0 { 0.0 } else { k as f64 / n as f64 }
    }
}

/// Empirical CDF over sorted lags, linear in the window with fewer than ten.
fn lag_cdf_sorted(sorted: &[f64], age: f64, window: f64) -> f64 {
    if age >= window {
        return 1.0;
    }
    if sorted.len() < 10 {
        return (age / window).clamp(0.0, 1.0);
    }
    sorted.partition_point(|l| *l <= age) as f64 / sorted.len() as f64
}
