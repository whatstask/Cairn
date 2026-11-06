//! A synthetic market for testing configs and the engine itself, offline.
//!
//! Each strategy has a hidden true conversion rate and guardrail rate. A
//! [`SimProposer`] drafts strategies whose hidden quality depends on the
//! operator used (local edits stay close to the parent; fresh ideas vary
//! widely; some are harmful). A market shock can make existing playbooks go
//! stale mid-run. Because the truth is known, a run reports how much the
//! engine gained over keeping the original strategy, how many bad
//! strategies it rolled back, and whether it ever overspent its budget.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use chrono::{DateTime, Duration, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use crate::engine::{EngineError, ManualClock, StrategyEngine};
use crate::event::StrategyEventKind;
use crate::propose::{ProposalRequest, Proposer, ProposerError};
use crate::rng::Rng;
use crate::types::{Operator, Stage, Strategy};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quality {
    pub rate: f64,
    pub guardrail_rate: f64,
    pub born_day: u32,
}

/// The hidden truth, keyed by strategy name.
#[derive(Default)]
pub struct Market {
    qualities: Mutex<HashMap<String, Quality>>,
    day: AtomicU64,
}

impl Market {
    pub fn insert(&self, name: &str, q: Quality) {
        self.qualities.lock().unwrap().insert(name.to_owned(), q);
    }

    pub fn get(&self, name: &str) -> Option<Quality> {
        self.qualities.lock().unwrap().get(name).copied()
    }

    fn today(&self) -> u32 {
        self.day.load(Ordering::Relaxed) as u32
    }
}

/// Drafts strategies with hidden quality drawn around their parents.
pub struct SimProposer {
    market: Arc<Market>,
    rng: Mutex<Rng>,
    counter: AtomicU64,
    base_rate: f64,
    toxic_rate: f64,
    guardrail_ok: f64,
    guardrail_bad: f64,
    /// Noise on the proposer's predicted lift (0 = a perfect oracle).
    prediction_noise: f64,
}

impl SimProposer {
    pub fn new(market: Arc<Market>, cfg: &SimConfig) -> Self {
        Self {
            market,
            rng: Mutex::new(Rng::seeded(cfg.seed ^ 0xA11CE)),
            counter: AtomicU64::new(0),
            base_rate: cfg.base_rate,
            toxic_rate: cfg.toxic_rate,
            guardrail_ok: cfg.guardrail_ok,
            guardrail_bad: cfg.guardrail_bad,
            prediction_noise: 0.15,
        }
    }
}

const SEGMENTS: [&str; 6] = ["fintech", "healthtech", "logistics", "retail", "saas", "manufacturing"];
const CHANNELS: [&str; 3] = ["email", "linkedin", "phone"];

#[async_trait]
impl Proposer for SimProposer {
    fn name(&self) -> String {
        "sim".into()
    }

    async fn propose(&self, req: &ProposalRequest) -> Result<Vec<Strategy>, ProposerError> {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut rng = self.rng.lock().unwrap();
        let parent = req.parents.first().map(|p| &p.strategy);
        let parent_rate = parent.and_then(|p| self.market.get(&p.name)).map(|q| q.rate).unwrap_or(self.base_rate);
        let (center, sigma) = match req.operator {
            Operator::Local => (parent_rate, 0.12),
            Operator::AxisShift => (parent_rate, 0.30),
            Operator::Crossover => {
                let other = req.parents.get(1).and_then(|p| self.market.get(&p.strategy.name)).map(|q| q.rate);
                (other.map_or(parent_rate, |o| (o + parent_rate) / 2.0), 0.20)
            }
            Operator::Fresh => (self.base_rate, 0.50),
        };
        let rate = (center * (sigma * rng.normal()).exp()).clamp(0.001, 0.9);
        let toxic = rng.uniform() < self.toxic_rate;
        let mut descriptor = parent.map(|p| p.descriptor.clone()).unwrap_or_default();
        match req.operator {
            Operator::AxisShift | Operator::Fresh => {
                descriptor.insert("segment".into(), SEGMENTS[rng.index(SEGMENTS.len())].into());
                descriptor.insert("channel".into(), CHANNELS[rng.index(CHANNELS.len())].into());
            }
            _ => {}
        }
        let name = format!("sim-{n}");
        self.market.insert(
            &name,
            Quality {
                rate,
                guardrail_rate: if toxic { self.guardrail_bad } else { self.guardrail_ok },
                born_day: self.market.today(),
            },
        );
        let true_lift = rate / parent_rate - 1.0;
        let strategy = Strategy {
            id: String::new(),
            name: name.clone(),
            descriptor,
            playbook: vec![
                format!("{name} opener h{:x}", rng.next_u64() >> 40),
                format!("proof point p{:x}", rng.next_u64() >> 40),
                format!("follow-up cadence c{:x}", rng.next_u64() >> 40),
            ],
            hypothesis: Some(format!("{:?} from {}", req.operator, parent.map_or("goal", |p| p.name.as_str()))),
            parents: Vec::new(),
            operator: None,
            proposer: None,
            predicted_lift: Some(true_lift + self.prediction_noise * rng.normal()),
        };
        Ok(vec![strategy])
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimConfig {
    pub days: u32,
    pub contacts_per_day: u32,
    pub base_rate: f64,
    pub mean_lag_hours: f64,
    /// From this day, every strategy born before it converts at
    /// `shock_factor` times its rate: the market moved on.
    pub shock_day: Option<u32>,
    pub shock_factor: f64,
    pub toxic_rate: f64,
    pub guardrail_ok: f64,
    pub guardrail_bad: f64,
    pub guardrail_event: String,
    pub success_event: String,
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            days: 180,
            contacts_per_day: 100,
            base_rate: 0.08,
            mean_lag_hours: 72.0,
            shock_day: Some(90),
            shock_factor: 0.5,
            toxic_rate: 0.15,
            guardrail_ok: 0.001,
            guardrail_bad: 0.03,
            guardrail_event: "spam_complaint".into(),
            success_event: "meeting_booked".into(),
            seed: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimReport {
    pub contacts: u64,
    pub successes: u64,
    /// Expected successes had every contact received the original strategy.
    pub baseline_expected: f64,
    /// Expected successes of the strategies actually assigned.
    pub engine_expected: f64,
    pub promotions: u32,
    pub rollbacks: u32,
    pub drift_alarms: u32,
    pub pauses: u32,
    pub proposals: u32,
    /// Contacts that went to a harmful strategy.
    pub toxic_contacts: u64,
    pub budget_respected: bool,
    pub final_incumbent: String,
    pub final_incumbent_rate: f64,
    pub best_rate_seen: f64,
}

impl SimReport {
    /// Relative gain of the engine over keeping the original strategy.
    pub fn lift(&self) -> f64 {
        self.engine_expected / self.baseline_expected - 1.0
    }
}

/// When simulated time starts. Create the program with the clock set here.
pub fn start() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 5, 9, 0, 0).unwrap()
}

/// Run a program through a simulated market. The engine must use `clock`,
/// and the program should be created at [`start`].
/// The simulated human approves every proposal that passes screening.
pub async fn run(
    engine: &StrategyEngine,
    clock: &ManualClock,
    program: &str,
    cfg: &SimConfig,
    market: Arc<Market>,
    proposer: &dyn Proposer,
) -> Result<SimReport, EngineError> {
    let start = start();
    clock.set(start);
    let mut rng = Rng::seeded(cfg.seed);
    let mut pending: BinaryHeap<Reverse<(DateTime<Utc>, String, String)>> = BinaryHeap::new();
    let state = engine.state(program).await?;
    let baseline_name = state.strategies[&state.baseline].strategy.name.clone();

    let quality_on = |name: &str, day: u32| -> Quality {
        let mut q = market.get(name).unwrap_or(Quality { rate: 0.0, guardrail_rate: 0.0, born_day: 0 });
        if let Some(shock) = cfg.shock_day {
            if day >= shock && q.born_day < shock {
                q.rate *= cfg.shock_factor;
            }
        }
        q
    };

    let mut report = SimReport {
        contacts: 0,
        successes: 0,
        baseline_expected: 0.0,
        engine_expected: 0.0,
        promotions: 0,
        rollbacks: 0,
        drift_alarms: 0,
        pauses: 0,
        proposals: 0,
        toxic_contacts: 0,
        budget_respected: true,
        final_incumbent: String::new(),
        final_incumbent_rate: 0.0,
        best_rate_seen: 0.0,
    };

    for day in 0..cfg.days {
        market.day.store(day as u64, Ordering::Relaxed);
        let day_start = start + Duration::days(day as i64);
        for i in 0..cfg.contacts_per_day {
            let now = day_start + Duration::seconds(i as i64 * 30);
            clock.set(now);
            flush(engine, program, &mut pending, now).await?;
            let a = engine.assign(program, &format!("d{day}-c{i}")).await?;
            if a.halt {
                continue;
            }
            report.contacts += 1;
            let q = quality_on(&a.strategy.name, day);
            report.engine_expected += q.rate;
            report.baseline_expected += quality_on(&baseline_name, day).rate;
            report.best_rate_seen = report.best_rate_seen.max(q.rate);
            if q.guardrail_rate > cfg.guardrail_ok {
                report.toxic_contacts += 1;
            }
            if rng.uniform() < q.rate {
                report.successes += 1;
                let at = now + Duration::seconds((rng.exponential(cfg.mean_lag_hours) * 3600.0) as i64);
                pending.push(Reverse((at, a.assignment_id.clone(), cfg.success_event.clone())));
            }
            if rng.uniform() < q.guardrail_rate {
                let at = now + Duration::hours(12);
                pending.push(Reverse((at, a.assignment_id.clone(), cfg.guardrail_event.clone())));
            }
        }
        let end = day_start + Duration::hours(23);
        clock.set(end);
        flush(engine, program, &mut pending, end).await?;
        let events = engine.tick(program).await?;
        for e in &events {
            match &e.kind {
                StrategyEventKind::StrategyPromoted { .. } => report.promotions += 1,
                StrategyEventKind::StrategyRolledBack { .. } => report.rollbacks += 1,
                StrategyEventKind::DriftDetected { .. } => report.drift_alarms += 1,
                StrategyEventKind::Paused { .. } => report.pauses += 1,
                _ => {}
            }
        }
        let state = engine.state(program).await?;
        if state.paused {
            engine.resume(program, "sim-operator").await?;
        }
        let open = state
            .strategies
            .values()
            .filter(|r| matches!(r.stage, Stage::AwaitingApproval | Stage::Approved | Stage::Canary))
            .count();
        let free = state.config.max_challengers.saturating_sub(open);
        if free > 0 {
            report.proposals += engine.generate(program, proposer, free).await?.len() as u32;
        }
        for r in engine.state(program).await?.strategies.values() {
            if r.stage == Stage::AwaitingApproval {
                engine.approve(program, &r.strategy.id, "sim-human", None).await?;
            }
        }
        let state = engine.state(program).await?;
        let status = engine.status(program).await?;
        if status.budget_spent > status.budget + 1e-9 && state.plays > 0 {
            report.budget_respected = false;
        }
    }
    let state = engine.state(program).await?;
    let inc = &state.strategies[&state.incumbent].strategy.name;
    report.final_incumbent = inc.clone();
    report.final_incumbent_rate = quality_on(inc, cfg.days.saturating_sub(1)).rate;
    Ok(report)
}

async fn flush(
    engine: &StrategyEngine,
    program: &str,
    pending: &mut BinaryHeap<Reverse<(DateTime<Utc>, String, String)>>,
    now: DateTime<Utc>,
) -> Result<(), EngineError> {
    while pending.peek().is_some_and(|Reverse((t, _, _))| *t <= now) {
        let Reverse((t, id, ev)) = pending.pop().expect("peeked");
        engine.report(program, &id, &ev, Some(t)).await?;
    }
    Ok(())
}

/// Hidden quality for the baseline strategy of a simulated program.
pub fn seed_baseline(market: &Market, name: &str, cfg: &SimConfig) {
    market.insert(name, Quality { rate: cfg.base_rate, guardrail_rate: cfg.guardrail_ok, born_day: 0 });
}

/// Descriptor used by simulated baselines.
pub fn baseline_descriptor() -> BTreeMap<String, String> {
    BTreeMap::from([("segment".into(), "saas".into()), ("channel".into(), "email".into())])
}
