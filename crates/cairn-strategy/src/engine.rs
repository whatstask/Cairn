use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Mutex;

use crate::archive::{self, TrackRecord};
use crate::event::{Shift, StrategyEvent, StrategyEventKind, verify_chain};
use crate::propose::{ArchiveEntry, ProposalRequest, Proposer};
use crate::rng::Rng;
use crate::screen::{Rules, Screen, Similarity, TokenJaccard, screen};
use crate::state::ProgramState;
use crate::stats::{
    BetaPrior, Counts, DropDetector, exceeds_evalue, fit_beta_prior, isotonic, prob_better, rate_bounds, spearman,
};
use crate::store::{StoreError, StrategyStore};
use crate::types::{Assignment, Operator, ProgramConfig, Stage, Strategy, StrategyId};

/// How long an allocation snapshot is reused before posteriors are refreshed.
const SNAPSHOT_TTL_MINUTES: i64 = 10;
/// Relative half-width the incumbent's rate must reach before it becomes the
/// drift reference.
const REFERENCE_PRECISION: f64 = 0.3;
/// Minimum time between updates of one calibrated parameter.
const CALIBRATION_INTERVAL_HOURS: i64 = 24;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("program `{0}` not found")]
    NotFound(String),
    #[error("program `{0}` already exists")]
    Exists(String),
    #[error("{0}")]
    Invalid(String),
    #[error("proposer failed: {0}")]
    Proposer(String),
    #[error("event log of `{program}` fails verification at seq {seq}")]
    Tampered { program: String, seq: u64 },
}

/// Where the engine gets the time. Tests and simulations drive a manual clock.
pub trait Clock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// A clock that only moves when told to.
pub struct ManualClock(std::sync::Mutex<DateTime<Utc>>);

impl ManualClock {
    pub fn new(start: DateTime<Utc>) -> Self {
        Self(std::sync::Mutex::new(start))
    }

    pub fn set(&self, t: DateTime<Utc>) {
        *self.0.lock().unwrap() = t;
    }

    pub fn advance(&self, d: Duration) {
        *self.0.lock().unwrap() += d;
    }
}

impl Clock for ManualClock {
    fn now(&self) -> DateTime<Utc> {
        *self.0.lock().unwrap()
    }
}

#[derive(Debug, Clone)]
struct ArmSnapshot {
    id: StrategyId,
    alpha: f64,
    beta: f64,
    mean: f64,
    lower: f64,
    share: f64,
    /// Traffic cap: ramps from the initial to the max share as clean
    /// guardrail evidence accumulates (progressive rollout).
    cap: f64,
}

#[derive(Debug, Clone)]
struct Snapshot {
    at: DateTime<Utc>,
    incumbent: ArmSnapshot,
    challengers: Vec<ArmSnapshot>,
}

struct Cached {
    state: ProgramState,
    snapshot: Option<Snapshot>,
}

/// Result of proposing a strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub id: StrategyId,
    pub stage: Stage,
}

/// The self-calibrating strategy engine.
///
/// The decision plane is deterministic Rust: Thompson sampling under a
/// conservative loss budget, anytime-valid promotion tests, an e-detector
/// for drift, guardrails with automatic rollback, and a persistent holdout.
/// Models only ever propose. Every decision and every self-calibrated
/// parameter change is an event in a hash-chained log, so the program's
/// whole history can be replayed and audited.
pub struct StrategyEngine {
    ids: Ids,
    store: Arc<dyn StrategyStore>,
    rules: Arc<dyn Screen>,
    similarity: Arc<dyn Similarity>,
    clock: Arc<dyn Clock>,
    programs: Mutex<HashMap<String, Cached>>,
}

impl StrategyEngine {
    pub fn new(store: Arc<dyn StrategyStore>) -> Self {
        Self {
            ids: Ids::Random,
            store,
            rules: Arc::new(Rules::default()),
            similarity: Arc::new(TokenJaccard),
            clock: Arc::new(SystemClock),
            programs: Mutex::new(HashMap::new()),
        }
    }

    pub fn rules(mut self, rules: Arc<dyn Screen>) -> Self {
        self.rules = rules;
        self
    }

    pub fn similarity(mut self, similarity: Arc<dyn Similarity>) -> Self {
        self.similarity = similarity;
        self
    }

    pub fn clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Generate ids from a seeded counter instead of random UUIDs, so a
    /// simulation (and every random draw seeded from an id) is reproducible.
    pub fn seeded_ids(mut self, seed: u64) -> Self {
        self.ids = Ids::Seeded { seed, next: std::sync::atomic::AtomicU64::new(0) };
        self
    }

    fn new_id(&self, prefix: &str) -> String {
        match &self.ids {
            Ids::Random => format!("{prefix}-{}", uuid::Uuid::now_v7().simple()),
            Ids::Seeded { seed, next } => {
                let n = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                format!("{prefix}-{seed:08x}{n:012x}")
            }
        }
    }

    pub fn now(&self) -> DateTime<Utc> {
        self.clock.now()
    }

    // ── program lifecycle ───────────────────────────────────────────────

    /// Create a program around the strategy agents use today. It becomes the
    /// incumbent and the frozen baseline for the holdout.
    pub async fn create_program(
        &self,
        name: &str,
        config: ProgramConfig,
        mut baseline: Strategy,
    ) -> Result<StrategyId, EngineError> {
        validate_config(&config)?;
        if name.is_empty() {
            return Err(EngineError::Invalid("program name is empty".into()));
        }
        let mut programs = self.programs.lock().await;
        if programs.contains_key(name) || !self.store.load(name).await?.is_empty() {
            return Err(EngineError::Exists(name.into()));
        }
        baseline.id = self.new_id("s");
        let id = baseline.id.clone();
        let event = StrategyEvent::new(name, 0, self.now(), StrategyEventKind::ProgramCreated { config, baseline }, "");
        self.store.append(&event).await?;
        let state = ProgramState::from_events(&[event]).expect("created event");
        programs.insert(name.into(), Cached { state, snapshot: None });
        Ok(id)
    }

    pub async fn list_programs(&self) -> Result<Vec<String>, EngineError> {
        Ok(self.store.list_programs().await?)
    }

    /// Current state (a clone), loading and verifying the log if needed.
    pub async fn state(&self, program: &str) -> Result<ProgramState, EngineError> {
        let mut programs = self.programs.lock().await;
        Ok(self.cached(&mut programs, program).await?.state.clone())
    }

    /// The full, hash-verified event log.
    pub async fn events(&self, program: &str) -> Result<Vec<StrategyEvent>, EngineError> {
        let events = self.store.load(program).await?;
        verify_chain(&events).map_err(|seq| EngineError::Tampered { program: program.into(), seq })?;
        Ok(events)
    }

    pub async fn update_config(&self, program: &str, config: ProgramConfig, by: &str) -> Result<(), EngineError> {
        validate_config(&config)?;
        let by = by.to_owned();
        self.commit(program, move |_, _| Ok((vec![StrategyEventKind::ConfigChanged { config, by }], ()))).await
    }

    /// Kill switch: every assignment answers `halt` until resumed.
    pub async fn pause(&self, program: &str, by: &str, reason: &str) -> Result<(), EngineError> {
        let (by, reason) = (by.to_owned(), reason.to_owned());
        self.commit(program, move |_, _| Ok((vec![StrategyEventKind::Paused { by, reason }], ()))).await
    }

    pub async fn resume(&self, program: &str, by: &str) -> Result<(), EngineError> {
        let by = by.to_owned();
        self.commit(program, move |_, _| Ok((vec![StrategyEventKind::Resumed { by }], ()))).await
    }

    // ── the hot path ────────────────────────────────────────────────────

    /// Which strategy to use for `subject` (a prospect, account or ticket).
    ///
    /// A subject keeps its strategy while that strategy is live, so
    /// multi-touch sequences stay consistent and outcomes stay attributable.
    pub async fn assign(&self, program: &str, subject: &str) -> Result<Assignment, EngineError> {
        let now = self.now();
        let mut programs = self.programs.lock().await;
        let cached = self.cached(&mut programs, program).await?;
        let state = &cached.state;

        if state.paused {
            let rec = &state.strategies[&state.incumbent];
            return Ok(Assignment {
                assignment_id: String::new(),
                strategy_id: state.incumbent.clone(),
                strategy: rec.strategy.clone(),
                propensity: 1.0,
                holdout: false,
                halt: true,
            });
        }

        if let Some(existing) = state.subjects.get(subject).and_then(|id| state.assignments.get(id).map(|a| (id, a))) {
            let (id, a) = existing;
            let live =
                state.stage(&a.strategy).is_some_and(Stage::is_live) || (a.holdout && a.strategy == state.baseline);
            if live {
                return Ok(Assignment {
                    assignment_id: id.clone(),
                    strategy_id: a.strategy.clone(),
                    strategy: state.strategies[&a.strategy].strategy.clone(),
                    propensity: a.propensity,
                    holdout: a.holdout,
                    halt: false,
                });
            }
        }

        let assignment_id = self.new_id("a");
        let holdout = state.config.holdout_fraction > 0.0
            && Rng::from_key(&format!("holdout:{program}:{subject}")).uniform() < state.config.holdout_fraction;

        let (strategy, propensity, loss, earned) = if holdout {
            (state.baseline.clone(), 1.0, 0.0, 0.0)
        } else {
            if cached.snapshot.as_ref().is_none_or(|s| now - s.at > Duration::minutes(SNAPSHOT_TTL_MINUTES)) {
                cached.snapshot = Some(snapshot(&cached.state, now));
            }
            let snap = cached.snapshot.as_ref().expect("snapshot");
            choose(&cached.state, snap, &assignment_id)
        };

        let kind = StrategyEventKind::Assigned {
            assignment_id: assignment_id.clone(),
            subject: subject.to_owned(),
            strategy: strategy.clone(),
            propensity,
            holdout,
            loss,
            earned,
            calibration_seq: cached.state.calibration_seq,
        };
        self.append_locked(cached, vec![kind], now).await?;
        let rec = &cached.state.strategies[&strategy];
        Ok(Assignment {
            assignment_id,
            strategy_id: strategy,
            strategy: rec.strategy.clone(),
            propensity,
            holdout,
            halt: false,
        })
    }

    /// Record an outcome for an assignment (e.g. `reply`, `meeting_booked`,
    /// `spam_complaint`). Idempotent: repeating the same event for the same
    /// assignment is a no-op and returns `false`.
    pub async fn report(
        &self,
        program: &str,
        assignment_id: &str,
        event: &str,
        occurred_at: Option<DateTime<Utc>>,
    ) -> Result<bool, EngineError> {
        let now = self.now();
        let (assignment_id, event) = (assignment_id.to_owned(), event.to_owned());
        self.commit(program, move |state, _| {
            let a = state
                .assignments
                .get(&assignment_id)
                .ok_or_else(|| EngineError::Invalid(format!("unknown assignment `{assignment_id}`")))?;
            if event.is_empty() {
                return Err(EngineError::Invalid("event name is empty".into()));
            }
            if a.event_at(&event).is_some() {
                return Ok((vec![], false));
            }
            let occurred_at = occurred_at.unwrap_or(now).max(a.at);
            Ok((vec![StrategyEventKind::OutcomeObserved { assignment_id, event, occurred_at }], true))
        })
        .await
    }

    // ── proposals and approvals ─────────────────────────────────────────

    /// Submit a strategy (from a human or a proposer). It is screened at once
    /// against the rules and the archive; scope changes wait for a human.
    pub async fn propose(&self, program: &str, mut strategy: Strategy) -> Result<Proposal, EngineError> {
        strategy.id = self.new_id("s");
        let rules = self.rules.clone();
        let similarity = self.similarity.clone();
        self.commit(program, move |state, _| {
            let (reasons, needs_approval) = screen(state, &strategy, rules.as_ref(), similarity.as_ref());
            let id = strategy.id.clone();
            let stage = match (reasons.is_empty(), needs_approval) {
                (false, _) => Stage::Rejected { reason: reasons.join("; ") },
                (true, true) => Stage::AwaitingApproval,
                (true, false) => Stage::Approved,
            };
            Ok((
                vec![
                    StrategyEventKind::StrategyProposed { strategy },
                    StrategyEventKind::StrategyScreened {
                        id: id.clone(),
                        passed: reasons.is_empty(),
                        reasons,
                        needs_approval,
                    },
                ],
                Proposal { id, stage },
            ))
        })
        .await
    }

    pub async fn approve(&self, program: &str, id: &str, by: &str, note: Option<String>) -> Result<(), EngineError> {
        let (id, by) = (id.to_owned(), by.to_owned());
        self.commit(program, move |state, _| {
            match state.stage(&id) {
                Some(Stage::AwaitingApproval) => {}
                Some(stage) => {
                    return Err(EngineError::Invalid(format!("`{id}` is not awaiting approval ({stage:?})")));
                }
                None => return Err(EngineError::Invalid(format!("unknown strategy `{id}`"))),
            }
            Ok((vec![StrategyEventKind::StrategyApproved { id, by, note }], ()))
        })
        .await
    }

    pub async fn reject(&self, program: &str, id: &str, by: &str, reason: &str) -> Result<(), EngineError> {
        let (id, by, reason) = (id.to_owned(), by.to_owned(), reason.to_owned());
        self.commit(program, move |state, _| {
            match state.stage(&id) {
                Some(Stage::AwaitingApproval | Stage::Approved) => {}
                Some(stage) => return Err(EngineError::Invalid(format!("`{id}` cannot be rejected ({stage:?})"))),
                None => return Err(EngineError::Invalid(format!("unknown strategy `{id}`"))),
            }
            Ok((vec![StrategyEventKind::StrategyRejected { id, by: Some(by), reason }], ()))
        })
        .await
    }

    /// Ask a proposer for `count` new strategies. Operators, parents and the
    /// proposer's context come from the archive and its track record; every
    /// draft then goes through [`propose`](Self::propose).
    pub async fn generate(
        &self,
        program: &str,
        proposer: &dyn Proposer,
        count: usize,
    ) -> Result<Vec<Proposal>, EngineError> {
        let now = self.now();
        let state = self.state(program).await?;
        let mix = archive::operator_mix(&state, now);
        let archive_entries: Vec<ArchiveEntry> = state
            .strategies
            .values()
            .map(|r| {
                let c = archive::lifetime_counts(&state, &r.strategy.id, now);
                ArchiveEntry {
                    strategy: r.strategy.clone(),
                    stage: r.stage.clone(),
                    rate: c.rate(),
                    outcomes: c.trials,
                }
            })
            .collect();
        let mut out = Vec::new();
        for i in 0..count {
            let mut rng = Rng::from_key(&format!("generate:{program}:{}:{i}", state.next_seq));
            let operator = archive::sample_operator(&mix, &mut rng);
            let parents = match operator {
                Operator::Fresh => Vec::new(),
                Operator::Crossover => archive::sample_parents(&state, 2, now, &mut rng),
                _ => archive::sample_parents(&state, 1, now, &mut rng),
            };
            let parent_entries: Vec<ArchiveEntry> =
                archive_entries.iter().filter(|e| parents.contains(&e.strategy.id)).cloned().collect();
            let request = ProposalRequest {
                goal: state.config.goal.clone(),
                operator,
                parents: parent_entries,
                archive: archive_entries.clone(),
                rules: self.rules.describe(),
                diagnosis: state.last_diagnosis.clone(),
                scope_axes: state.config.scope_axes.clone(),
            };
            let drafts = proposer.propose(&request).await.map_err(|e| EngineError::Proposer(e.to_string()))?;
            for mut draft in drafts {
                if draft.parents.is_empty() {
                    draft.parents = parents.clone();
                }
                draft.operator = Some(operator);
                draft.proposer = Some(proposer.name());
                out.push(self.propose(program, draft).await?);
            }
        }
        Ok(out)
    }

    // ── the decision loop ───────────────────────────────────────────────

    /// Run every scheduled decision: self-calibration, canary starts,
    /// guardrails, drift detection, promotion and retirement, and proposal
    /// requests. Call it on a schedule (hourly is plenty). Returns the events
    /// it recorded.
    pub async fn tick(&self, program: &str) -> Result<Vec<StrategyEvent>, EngineError> {
        let now = self.now();
        let mut programs = self.programs.lock().await;
        let cached = self.cached(&mut programs, program).await?;
        let mut recorded = Vec::new();
        // Each phase sees the effects of the previous one.
        for phase in [calibrate as Phase, start_canaries, guardrails, drift, decide, request_proposals] {
            let kinds = phase(&cached.state, now);
            if !kinds.is_empty() {
                recorded.extend(self.append_locked(cached, kinds, now).await?);
            }
        }
        if !recorded.is_empty() {
            cached.snapshot = None;
        }
        Ok(recorded)
    }

    /// A readable summary of the program.
    pub async fn status(&self, program: &str) -> Result<ProgramStatus, EngineError> {
        let now = self.now();
        let mut programs = self.programs.lock().await;
        let cached = self.cached(&mut programs, program).await?;
        Ok(status(&cached.state, now))
    }

    // ── plumbing ────────────────────────────────────────────────────────

    async fn cached<'a>(
        &self,
        programs: &'a mut HashMap<String, Cached>,
        program: &str,
    ) -> Result<&'a mut Cached, EngineError> {
        if !programs.contains_key(program) {
            let events = self.store.load(program).await?;
            verify_chain(&events).map_err(|seq| EngineError::Tampered { program: program.into(), seq })?;
            let state = ProgramState::from_events(&events).ok_or_else(|| EngineError::NotFound(program.into()))?;
            programs.insert(program.into(), Cached { state, snapshot: None });
        }
        Ok(programs.get_mut(program).expect("inserted"))
    }

    async fn append_locked(
        &self,
        cached: &mut Cached,
        kinds: Vec<StrategyEventKind>,
        now: DateTime<Utc>,
    ) -> Result<Vec<StrategyEvent>, EngineError> {
        let mut out = Vec::new();
        for kind in kinds {
            let structural =
                !matches!(kind, StrategyEventKind::Assigned { .. } | StrategyEventKind::OutcomeObserved { .. });
            let event =
                StrategyEvent::new(&cached.state.name, cached.state.next_seq, now, kind, &cached.state.last_hash);
            self.store.append(&event).await?;
            cached.state.apply(&event);
            if structural {
                cached.snapshot = None;
            }
            out.push(event);
        }
        Ok(out)
    }

    /// Validate against current state and append. On a sequence conflict
    /// (another engine instance wrote first) the cached state is dropped, so
    /// the caller's retry sees the fresh history.
    async fn commit<R, F>(&self, program: &str, f: F) -> Result<R, EngineError>
    where
        F: FnOnce(&ProgramState, DateTime<Utc>) -> Result<(Vec<StrategyEventKind>, R), EngineError>,
    {
        let now = self.now();
        let mut programs = self.programs.lock().await;
        let cached = self.cached(&mut programs, program).await?;
        let (kinds, result) = f(&cached.state, now)?;
        match self.append_locked(cached, kinds, now).await {
            Ok(_) => Ok(result),
            Err(e) => {
                if matches!(e, EngineError::Store(StoreError::Conflict { .. })) {
                    programs.remove(program);
                }
                Err(e)
            }
        }
    }
}

type Phase = fn(&ProgramState, DateTime<Utc>) -> Vec<StrategyEventKind>;

enum Ids {
    Random,
    Seeded { seed: u64, next: std::sync::atomic::AtomicU64 },
}

fn validate_config(c: &ProgramConfig) -> Result<(), EngineError> {
    let unit = |name: &str, v: f64| {
        if (0.0..=1.0).contains(&v) { Ok(()) } else { Err(EngineError::Invalid(format!("`{name}` must be in [0, 1]"))) }
    };
    unit("holdout_fraction", c.holdout_fraction)?;
    unit("conservative_alpha", c.conservative_alpha)?;
    unit("exploration_floor", c.exploration_floor)?;
    unit("canary_max_share", c.canary_max_share)?;
    unit("decision_alpha", c.decision_alpha)?;
    if c.success_event.is_empty() {
        return Err(EngineError::Invalid("`success_event` is empty".into()));
    }
    if c.maturity_bounds_hours.0 <= 0.0 || c.maturity_bounds_hours.0 > c.maturity_bounds_hours.1 {
        return Err(EngineError::Invalid("`maturity_bounds_hours` must be positive and ordered".into()));
    }
    if c.drift_alarm_days < 1.0 {
        return Err(EngineError::Invalid("`drift_alarm_days` must be at least 1".into()));
    }
    Ok(())
}

// ── allocation ──────────────────────────────────────────────────────────

fn arm_snapshot(state: &ProgramState, id: &str, now: DateTime<Utc>) -> ArmSnapshot {
    let rec = &state.strategies[id];
    let since = rec.live_at.unwrap_or(state.created_at);
    let (alpha, beta) = state.prior.posterior(state.allocation_counts(id, now));
    let matured = state.matured_counts(id, since, now, false);
    let (lower, _) = rate_bounds(matured, state.config.decision_alpha);
    ArmSnapshot {
        id: id.to_owned(),
        alpha,
        beta,
        mean: alpha / (alpha + beta),
        lower,
        share: if id == state.incumbent { 1.0 } else { state.share(id, since) },
        cap: canary_cap(state, id, since, now),
    }
}

fn canary_cap(state: &ProgramState, id: &str, since: DateTime<Utc>, now: DateTime<Utc>) -> f64 {
    let cfg = &state.config;
    let (start, max) = (cfg.canary_initial_share.min(cfg.canary_max_share), cfg.canary_max_share);
    // Only clean evidence ramps traffic up: a guardrail running above its
    // limit holds the canary at its initial share until it is rolled back.
    let evidence = cfg
        .guardrails
        .iter()
        .map(|g| {
            let c = state.guardrail_counts(id, &g.event, since, now);
            if c.rate().is_some_and(|r| r > g.max_rate) { 0.0 } else { c.trials }
        })
        .fold(f64::INFINITY, f64::min);
    if evidence.is_infinite() {
        return max;
    }
    start + (max - start) * (evidence / (2.0 * cfg.min_samples)).min(1.0)
}

fn snapshot(state: &ProgramState, now: DateTime<Utc>) -> Snapshot {
    Snapshot {
        at: now,
        incumbent: arm_snapshot(state, &state.incumbent, now),
        challengers: state.challengers().iter().map(|id| arm_snapshot(state, id, now)).collect(),
    }
}

/// Thompson sampling with forced exploration, gated by the canary cap and the
/// conservative budget. Returns `(strategy, propensity, loss, earned)`.
fn choose(state: &ProgramState, snap: &Snapshot, assignment_id: &str) -> (StrategyId, f64, f64, f64) {
    let cfg = &state.config;
    let inc = &snap.incumbent;
    let earned = cfg.conservative_alpha * inc.mean;
    if snap.challengers.is_empty() {
        return (inc.id.clone(), 1.0, 0.0, earned);
    }
    // Pessimistic cost of a challenger play, and whether the budget allows it.
    let loss = |arm: &ArmSnapshot| (inc.mean - arm.lower).max(0.0);
    let budget = state.budget_earned + earned;
    let allowed = |arm: &ArmSnapshot| arm.share < arm.cap && state.loss_spent + loss(arm) <= budget;

    let draw = |rng: &mut Rng| -> usize {
        let forced = cfg.exploration_floor * snap.challengers.len() as f64;
        let pick = if rng.uniform() < forced {
            1 + rng.index(snap.challengers.len())
        } else {
            let mut best = (0, rng.beta(inc.alpha, inc.beta));
            for (i, c) in snap.challengers.iter().enumerate() {
                let x = rng.beta(c.alpha, c.beta);
                if x > best.1 {
                    best = (i + 1, x);
                }
            }
            best.0
        };
        if pick > 0 && !allowed(&snap.challengers[pick - 1]) { 0 } else { pick }
    };

    let chosen = draw(&mut Rng::from_key(assignment_id));
    let samples = cfg.propensity_samples.max(1);
    let mut prop_rng = Rng::from_key(&format!("{assignment_id}:propensity"));
    let hits = (0..samples).filter(|_| draw(&mut prop_rng) == chosen).count();
    let propensity = (hits.max(1) as f64) / samples as f64;
    if chosen == 0 {
        (inc.id.clone(), propensity, 0.0, earned)
    } else {
        let arm = &snap.challengers[chosen - 1];
        (arm.id.clone(), propensity, loss(arm), earned)
    }
}

// ── tick phases ─────────────────────────────────────────────────────────

fn due(state: &ProgramState, param: &str, now: DateTime<Utc>) -> bool {
    state.last_calibrated.get(param).is_none_or(|t| now - *t >= Duration::hours(CALIBRATION_INTERVAL_HOURS))
}

fn bounded_step(old: f64, new: f64, max_step: f64) -> f64 {
    new.clamp(old / max_step, old * max_step)
}

/// Self-calibration. Only matured outcomes are used; every parameter moves
/// at most `calibration_max_step` per day and is journaled with its evidence.
/// The metric contract, guardrails and approval rules are never touched.
fn calibrate(state: &ProgramState, now: DateTime<Utc>) -> Vec<StrategyEventKind> {
    let cfg = &state.config;
    let mut out = Vec::new();

    // Maturity window from observed conversion lags, using only a cohort
    // old enough that its conversions have arrived (no right-censoring).
    if due(state, "maturity_hours", now) {
        let old = state.maturity_hours;
        let cohort_age = cfg.maturity_bounds_hours.1.max(2.0 * old);
        let mut lags = state.success_lags_older_than(Some((cohort_age, now)));
        if lags.len() >= 30 {
            lags.sort_by(f64::total_cmp);
            let q90 = lags[(lags.len() as f64 * 0.9) as usize - 1];
            let new = bounded_step(old, q90 * 1.25, cfg.calibration_max_step)
                .clamp(cfg.maturity_bounds_hours.0, cfg.maturity_bounds_hours.1);
            if (new - old).abs() / old > 0.2 {
                out.push(StrategyEventKind::Calibrated {
                    param: "maturity_hours".into(),
                    old: json!(old),
                    new: json!(new),
                    evidence: format!(
                        "{} conversion lags from assignments older than {cohort_age:.0}h, p90 {q90:.1}h",
                        lags.len()
                    ),
                });
            }
        }
    }

    // Guardrail window from how fast guardrail events arrive.
    if due(state, "guardrail_window_hours", now) && !cfg.guardrails.is_empty() {
        let old = state.guardrail_window_hours;
        let mut lags = state.guardrail_lags(24.0 * 14.0, now);
        if lags.len() >= 20 {
            lags.sort_by(f64::total_cmp);
            let q95 = lags[(lags.len() as f64 * 0.95) as usize - 1];
            let new = bounded_step(old, q95 * 1.25, cfg.calibration_max_step).clamp(6.0, 24.0 * 14.0);
            if (new - old).abs() / old > 0.2 {
                out.push(StrategyEventKind::Calibrated {
                    param: "guardrail_window_hours".into(),
                    old: json!(old),
                    new: json!(new),
                    evidence: format!("{} guardrail event lags, p95 {q95:.1}h", lags.len()),
                });
            }
        }
    }

    // Prior by empirical Bayes across strategies that have had traffic.
    if due(state, "prior", now) {
        let arms: Vec<Counts> = state
            .strategies
            .values()
            .filter(|r| r.live_at.is_some())
            .map(|r| state.matured_counts(&r.strategy.id, r.live_at.unwrap_or(state.created_at), now, false))
            .filter(|c| c.trials >= cfg.min_samples)
            .collect();
        let (lo, hi) = cfg.prior_strength_bounds;
        if let Some(fit) = fit_beta_prior(&arms, lo, hi) {
            let old = state.prior;
            let strength = bounded_step(old.strength(), fit.strength(), cfg.calibration_max_step).clamp(lo, hi);
            let new = BetaPrior { alpha: fit.mean() * strength, beta: (1.0 - fit.mean()) * strength };
            let mean_moved = (new.mean() - old.mean()).abs() / old.mean().max(1e-6) > 0.1;
            let strength_moved = (new.strength() - old.strength()).abs() / old.strength() > 0.2;
            if mean_moved || strength_moved {
                out.push(StrategyEventKind::Calibrated {
                    param: "prior".into(),
                    old: json!(old),
                    new: json!(new),
                    evidence: format!("empirical Bayes over {} strategies with matured outcomes", arms.len()),
                });
            }
        }
    }

    // Calibrate proposers' predicted lift against realized lift.
    if due(state, "prediction_fit", now) {
        let pairs: Vec<(f64, f64)> = state
            .strategies
            .values()
            .filter_map(|r| {
                let predicted = r.strategy.predicted_lift?;
                let start = r.live_at?;
                if r.stage == Stage::Canary || r.strategy.id == state.baseline {
                    return None;
                }
                let end = r.stage_at;
                let ch = state.matured_counts(&r.strategy.id, start, now, false);
                let others = state
                    .assignments
                    .values()
                    .filter(|a| a.strategy != r.strategy.id && !a.holdout && a.at >= start && a.at <= end)
                    .fold(Counts::default(), |mut c, a| {
                        if let Some(s) = a.matured(&state.config.success_event, state.maturity_hours, now) {
                            c.add(Counts::new(1.0, s as u8 as f64));
                        }
                        c
                    });
                let realized = ch.rate()? / others.rate().filter(|r| *r > 0.0)? - 1.0;
                (ch.trials >= cfg.min_samples).then_some((predicted, realized))
            })
            .collect();
        if pairs.len() >= 5 {
            let fit = isotonic(&pairs);
            let trust = spearman(&pairs).unwrap_or(0.0).max(0.0);
            if fit != state.prediction_fit {
                out.push(StrategyEventKind::Calibrated {
                    param: "prediction_fit".into(),
                    old: json!(state.prediction_fit),
                    new: json!(fit),
                    evidence: format!("isotonic fit of predicted vs realized lift over {} strategies", pairs.len()),
                });
            }
            if (trust - state.prediction_trust).abs() > 0.05 {
                out.push(StrategyEventKind::Calibrated {
                    param: "prediction_trust".into(),
                    old: json!(state.prediction_trust),
                    new: json!(trust),
                    evidence: format!("Spearman rank correlation of predicted vs realized lift, n = {}", pairs.len()),
                });
            }
        }
    }
    out
}

fn start_canaries(state: &ProgramState, _now: DateTime<Utc>) -> Vec<StrategyEventKind> {
    if state.paused {
        return Vec::new();
    }
    let free = state.config.max_challengers.saturating_sub(state.challengers().len());
    state.approved_queue().into_iter().take(free).map(|id| StrategyEventKind::CanaryStarted { id }).collect()
}

fn guardrails(state: &ProgramState, now: DateTime<Utc>) -> Vec<StrategyEventKind> {
    let cfg = &state.config;
    let mut out = Vec::new();
    let mut live: Vec<StrategyId> = state.challengers();
    if !state.paused {
        live.push(state.incumbent.clone());
    }
    for id in live {
        let rec = &state.strategies[&id];
        let since = rec.live_at.unwrap_or(rec.stage_at);
        for g in &cfg.guardrails {
            let c = state.guardrail_counts(&id, &g.event, since, now);
            if c.trials < 1.0 {
                continue;
            }
            let e = exceeds_evalue(c, g.max_rate);
            if e >= 1.0 / cfg.decision_alpha {
                let (lower, _) = rate_bounds(c, cfg.decision_alpha);
                out.push(StrategyEventKind::GuardrailBreached {
                    strategy: id.clone(),
                    event: g.event.clone(),
                    lower_bound: lower,
                    limit: g.max_rate,
                });
                let reason = format!(
                    "`{}` rate {:.4} is above {} (e-value {e:.0}, {} outcomes)",
                    g.event,
                    c.rate().unwrap_or(0.0),
                    g.max_rate,
                    c.trials as u64
                );
                if id == state.incumbent {
                    out.push(StrategyEventKind::Paused { by: "engine".into(), reason });
                } else {
                    out.push(StrategyEventKind::StrategyRolledBack { id: id.clone(), reason });
                }
                break;
            }
        }
    }
    out
}

/// Outcome-rate shifts for `arm` around the most likely change point since
/// `from` (found by GLR), largest first. Only matured assignments count, so
/// every event had the same time to happen.
fn diagnose(state: &ProgramState, arm: &str, from: DateTime<Utc>, now: DateTime<Utc>) -> Vec<Shift> {
    let buckets = state.matured_buckets(arm, from, now);
    let split_day = crate::stats::glr_change(&buckets, 0.5, 10.0).map(|c| c.split).unwrap_or(buckets.len() / 2);
    let split = from + Duration::days(split_day as i64);
    let mut shifts = Vec::new();
    for event in state.event_names() {
        let (mut b, mut a) = (Counts::default(), Counts::default());
        for rec in state
            .assignments
            .values()
            .filter(|r| r.strategy == arm && !r.holdout && r.at >= from && r.age_hours(now) >= state.maturity_hours)
        {
            let hit = Counts::new(1.0, rec.matured(&event, state.maturity_hours, now).unwrap_or(false) as u8 as f64);
            if rec.at < split { b.add(hit) } else { a.add(hit) }
        }
        let (Some(pb), Some(pa)) = (b.rate(), a.rate()) else { continue };
        let pooled = (b.successes + a.successes) / (b.trials + a.trials);
        let se = (pooled * (1.0 - pooled) * (1.0 / b.trials + 1.0 / a.trials)).sqrt();
        let z = if se > 0.0 { (pa - pb) / se } else { 0.0 };
        shifts.push(Shift { event, before: pb, after: pa, z });
    }
    shifts.sort_by(|x, y| y.z.abs().total_cmp(&x.z.abs()));
    shifts
}

fn drift(state: &ProgramState, now: DateTime<Utc>) -> Vec<StrategyEventKind> {
    if state.paused {
        return Vec::new();
    }
    let cfg = &state.config;
    let m = &state.monitor;
    if m.strategy != state.incumbent {
        return Vec::new();
    }
    match m.reference {
        None => {
            let c = state.matured_counts(&m.strategy, m.from, now, false);
            if c.trials < cfg.min_samples {
                return Vec::new();
            }
            // A lower confidence bound as reference: if anything, it
            // under-states the rate, which only makes false alarms rarer.
            let (lower, upper) = rate_bounds(c, cfg.decision_alpha);
            // Wait until the rate is pinned down to within about +-30%; a
            // loose reference would hide real drops.
            let rate = c.rate().unwrap_or(0.0);
            if lower <= 0.0 || (upper - lower) / 2.0 > REFERENCE_PRECISION * rate {
                return Vec::new();
            }
            vec![StrategyEventKind::DriftReferenceSet { strategy: m.strategy.clone(), reference: lower, from: now }]
        }
        Some(reference) => {
            let buckets = state.matured_buckets(&m.strategy, m.from, now);
            let mut detector = DropDetector::new(reference);
            let mut value = 0.0;
            for b in &buckets {
                value = detector.update(*b);
            }
            if value < cfg.drift_alarm_days {
                return Vec::new();
            }
            let recent = buckets.iter().fold(Counts::default(), |mut a, b| {
                a.add(*b);
                a
            });
            vec![
                StrategyEventKind::DriftDetected {
                    strategy: m.strategy.clone(),
                    reference,
                    recent: recent.rate().unwrap_or(0.0),
                    detector: value,
                    diagnosis: diagnose(state, &m.strategy, (m.from - Duration::days(30)).max(state.created_at), now),
                },
                StrategyEventKind::ProposalsRequested { reason: "incumbent performance dropped".into() },
            ]
        }
    }
}

/// Promote a challenger only when an anytime-valid test shows it beats the
/// incumbent over the same period and it breaches no guardrail. Retire it
/// once it is unlikely ever to win, so its slot goes to a new idea: that
/// call can only cost an opportunity, never harm, so a posterior
/// probability is enough.
fn decide(state: &ProgramState, now: DateTime<Utc>) -> Vec<StrategyEventKind> {
    if state.paused {
        return Vec::new();
    }
    let cfg = &state.config;
    // Union bound: most of the error budget goes to the challenger's bound,
    // since the incumbent usually has far more data.
    let (alpha_ch, alpha_inc) = (0.8 * cfg.decision_alpha, 0.2 * cfg.decision_alpha);
    let mut out = Vec::new();
    let mut promoted = false;
    for id in state.challengers() {
        let since = state.strategies[&id].live_at.unwrap_or(state.created_at);
        let ch = state.matured_counts(&id, since, now, false);
        let inc = state.matured_counts(&state.incumbent, since, now, false);
        if ch.trials < cfg.min_samples || inc.trials < cfg.min_samples {
            if now - since >= Duration::days(cfg.max_canary_days as i64) {
                out.push(StrategyEventKind::StrategyRetired { id, reason: "too little traffic to judge".into() });
            }
            continue;
        }
        let (cl, _) = rate_bounds(ch, alpha_ch);
        let (_, iu) = rate_bounds(inc, alpha_inc);
        let guardrails_ok = cfg.guardrails.iter().all(|g| {
            let c = state.guardrail_counts(&id, &g.event, since, now);
            c.rate().is_none_or(|r| r <= g.max_rate)
        });
        let p_better = prob_better(ch, inc);
        if !promoted && cl > iu && guardrails_ok {
            out.push(StrategyEventKind::StrategyPromoted {
                id: id.clone(),
                replaced: state.incumbent.clone(),
                challenger_rate: ch.rate().unwrap_or(0.0),
                incumbent_rate: inc.rate().unwrap_or(0.0),
            });
            promoted = true;
        } else if p_better < 0.05 {
            out.push(StrategyEventKind::StrategyRetired {
                id,
                reason: format!("unlikely to beat the incumbent (P = {p_better:.3})"),
            });
        } else if ch.trials >= cfg.max_samples || now - since >= Duration::days(cfg.max_canary_days as i64) {
            out.push(StrategyEventKind::StrategyRetired {
                id,
                reason: format!("no significant lift after {} outcomes", ch.trials as u64),
            });
        }
    }
    out
}

fn request_proposals(state: &ProgramState, now: DateTime<Utc>) -> Vec<StrategyEventKind> {
    if state.paused {
        return Vec::new();
    }
    let open = state
        .strategies
        .values()
        .filter(|r| matches!(r.stage, Stage::AwaitingApproval | Stage::Approved | Stage::Canary))
        .count();
    let recently = state.last_proposal_request.is_some_and(|t| now - t < Duration::days(7));
    if open < state.config.max_challengers && !recently {
        vec![StrategyEventKind::ProposalsRequested { reason: "challenger slots are free".into() }]
    } else {
        Vec::new()
    }
}

// ── status ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmStatus {
    pub id: StrategyId,
    pub name: String,
    pub stage: Stage,
    pub matured: Counts,
    pub rate: Option<f64>,
    /// Anytime-valid interval at the program's decision alpha.
    pub interval: (f64, f64),
    pub share: f64,
    pub guardrails: Vec<(String, f64)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HoldoutStatus {
    pub baseline_rate: Option<f64>,
    pub baseline_outcomes: f64,
    pub treated_rate: Option<f64>,
    pub treated_outcomes: f64,
    /// Relative lift of everything the engine did over the frozen baseline.
    pub lift: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramStatus {
    pub program: String,
    pub paused: bool,
    pub incumbent: StrategyId,
    pub baseline: StrategyId,
    pub arms: Vec<ArmStatus>,
    pub awaiting_approval: Vec<Strategy>,
    pub approved_queue: Vec<StrategyId>,
    pub holdout: HoldoutStatus,
    pub maturity_hours: f64,
    pub prior: BetaPrior,
    pub prediction_trust: f64,
    pub drift_reference: Option<f64>,
    pub budget_spent: f64,
    pub budget: f64,
    pub operators: Vec<(Operator, f64, TrackRecord)>,
    pub proposers: Vec<(String, TrackRecord)>,
    pub stagnating: bool,
    pub plays: u64,
    pub events: u64,
}

fn status(state: &ProgramState, now: DateTime<Utc>) -> ProgramStatus {
    let cfg = &state.config;
    let mut live: Vec<StrategyId> = vec![state.incumbent.clone()];
    live.extend(state.challengers());
    let arms = live
        .iter()
        .map(|id| {
            let rec = &state.strategies[id];
            let since = rec.live_at.unwrap_or(state.created_at);
            let matured = state.matured_counts(id, since, now, false);
            ArmStatus {
                id: id.clone(),
                name: rec.strategy.name.clone(),
                stage: rec.stage.clone(),
                matured,
                rate: matured.rate(),
                interval: rate_bounds(matured, cfg.decision_alpha),
                share: state.share(id, since),
                guardrails: cfg
                    .guardrails
                    .iter()
                    .map(|g| (g.event.clone(), state.guardrail_counts(id, &g.event, since, now).rate().unwrap_or(0.0)))
                    .collect(),
            }
        })
        .collect();
    let baseline = state.matured_counts(&state.baseline, state.created_at, now, true);
    let treated = state.assignments.values().filter(|a| !a.holdout).fold(Counts::default(), |mut c, a| {
        if let Some(s) = a.matured(&cfg.success_event, state.maturity_hours, now) {
            c.add(Counts::new(1.0, s as u8 as f64));
        }
        c
    });
    let lift = match (baseline.rate(), treated.rate()) {
        (Some(b), Some(t)) if b > 0.0 => Some(t / b - 1.0),
        _ => None,
    };
    let mix = archive::operator_mix(state, now);
    let records = archive::operator_records(state);
    ProgramStatus {
        program: state.name.clone(),
        paused: state.paused,
        incumbent: state.incumbent.clone(),
        baseline: state.baseline.clone(),
        arms,
        awaiting_approval: state
            .strategies
            .values()
            .filter(|r| r.stage == Stage::AwaitingApproval)
            .map(|r| r.strategy.clone())
            .collect(),
        approved_queue: state.approved_queue(),
        holdout: HoldoutStatus {
            baseline_rate: baseline.rate(),
            baseline_outcomes: baseline.trials,
            treated_rate: treated.rate(),
            treated_outcomes: treated.trials,
            lift,
        },
        maturity_hours: state.maturity_hours,
        prior: state.prior,
        prediction_trust: state.prediction_trust,
        drift_reference: state.monitor.reference,
        budget_spent: state.loss_spent,
        budget: state.budget_earned,
        operators: mix.iter().map(|(o, p)| (*o, *p, records.get(o).cloned().unwrap_or_default())).collect(),
        proposers: archive::proposer_records(state).into_iter().collect(),
        stagnating: archive::stagnating(state, now),
        plays: state.plays,
        events: state.next_seq,
    }
}
