use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use cairn_runtime::testing::{ScriptedModel, text_response};
use cairn_strategy::sim::{self, Market, Quality, SimConfig, SimProposer};
use cairn_strategy::*;
use chrono::{Duration, NaiveDate, TimeZone, Utc};

fn engine(clock: Arc<ManualClock>) -> StrategyEngine {
    StrategyEngine::new(Arc::new(InMemoryStore::new())).clock(clock).seeded_ids(42)
}

fn clock() -> Arc<ManualClock> {
    Arc::new(ManualClock::new(Utc.with_ymd_and_hms(2026, 1, 5, 9, 0, 0).unwrap()))
}

fn config(cfg: &SimConfig) -> ProgramConfig {
    ProgramConfig {
        goal: "book meetings with B2B buyers".into(),
        success_event: cfg.success_event.clone(),
        guardrails: vec![Guardrail { event: cfg.guardrail_event.clone(), max_rate: 0.005 }],
        ..Default::default()
    }
}

fn baseline() -> Strategy {
    let mut s = Strategy::new("baseline", vec!["Open with the customer's compliance pain".into()]);
    s.descriptor = sim::baseline_descriptor();
    s
}

/// Hands out a fixed list of drafts, then nothing.
struct FixedProposer(Mutex<Vec<Strategy>>);

#[async_trait]
impl Proposer for FixedProposer {
    fn name(&self) -> String {
        "fixed".into()
    }

    async fn propose(&self, _req: &ProposalRequest) -> Result<Vec<Strategy>, ProposerError> {
        Ok(self.0.lock().unwrap().pop().into_iter().collect())
    }
}

fn local_edit(name: &str) -> Strategy {
    let mut s = Strategy::new(name, vec![format!("{name}: lead with a two-line case study")]);
    s.descriptor = sim::baseline_descriptor();
    s
}

async fn run_fixed(
    cfg: &SimConfig,
    drafts: Vec<(Strategy, Quality)>,
    program_cfg: ProgramConfig,
) -> (StrategyEngine, sim::SimReport) {
    let clock = clock();
    let engine = engine(clock.clone());
    let market = Arc::new(Market::default());
    sim::seed_baseline(&market, "baseline", cfg);
    for (s, q) in &drafts {
        market.insert(&s.name, *q);
    }
    engine.create_program("p", program_cfg, baseline()).await.unwrap();
    let proposer = FixedProposer(Mutex::new(drafts.into_iter().rev().map(|d| d.0).collect()));
    let report = sim::run(&engine, &clock, "p", cfg, market, &proposer).await.unwrap();
    (engine, report)
}

#[tokio::test]
async fn assignments_are_sticky_and_the_holdout_is_stable() {
    let clock = clock();
    let engine = engine(clock.clone());
    let cfg = ProgramConfig { holdout_fraction: 0.2, success_event: "won".into(), ..Default::default() };
    engine.create_program("p", cfg, baseline()).await.unwrap();
    let mut holdout = 0;
    for i in 0..2000 {
        let a = engine.assign("p", &format!("s{i}")).await.unwrap();
        holdout += a.holdout as u32;
        let again = engine.assign("p", &format!("s{i}")).await.unwrap();
        assert_eq!(a.assignment_id, again.assignment_id);
    }
    assert!((340..=460).contains(&holdout), "holdout {holdout}/2000");
}

#[tokio::test]
async fn screening_rejects_duplicates_and_rule_breaks_and_gates_scope_changes() {
    let clock = clock();
    let rules = Rules { banned_terms: vec!["guaranteed".into()], ..Default::default() };
    let engine = engine(clock.clone()).rules(Arc::new(rules));
    let cfg = ProgramConfig { auto_approve_local: true, success_event: "won".into(), ..Default::default() };
    let base_id = engine.create_program("p", cfg, baseline()).await.unwrap();

    let dup = engine.propose("p", baseline()).await.unwrap();
    assert!(matches!(dup.stage, Stage::Rejected { ref reason } if reason.contains("duplicates")), "{dup:?}");

    let mut banned = local_edit("banned");
    banned.playbook.push("Guaranteed results in 7 days".into());
    let p = engine.propose("p", banned).await.unwrap();
    assert!(matches!(p.stage, Stage::Rejected { ref reason } if reason.contains("banned")), "{p:?}");

    let mut edit = local_edit("edit");
    edit.parents = vec![base_id.clone()];
    edit.operator = Some(Operator::Local);
    assert_eq!(engine.propose("p", edit).await.unwrap().stage, Stage::Approved);

    let mut shift = local_edit("new market");
    shift.parents = vec![base_id];
    shift.operator = Some(Operator::Local);
    shift.descriptor.insert("segment".into(), "healthtech".into());
    let p = engine.propose("p", shift).await.unwrap();
    assert_eq!(p.stage, Stage::AwaitingApproval, "a segment change always needs a human");

    engine.approve("p", &p.id, "maria", None).await.unwrap();
    assert!(engine.approve("p", &p.id, "maria", None).await.is_err(), "cannot approve twice");
}

#[tokio::test]
async fn a_better_challenger_is_promoted_and_the_log_replays() {
    let cfg = SimConfig { days: 120, shock_day: None, ..Default::default() };
    let good = (local_edit("good"), Quality { rate: 0.16, guardrail_rate: 0.001, born_day: 0 });
    let (engine, report) = run_fixed(&cfg, vec![good], config(&cfg)).await;
    assert_eq!(report.promotions, 1, "{report:?}");
    assert_eq!(report.final_incumbent, "good");
    assert!(report.budget_respected);

    let events = engine.events("p").await.unwrap();
    verify_chain(&events).unwrap();
    let replayed = ProgramState::from_events(&events).unwrap();
    assert_eq!(replayed, engine.state("p").await.unwrap());
}

#[tokio::test]
async fn an_equal_challenger_is_not_promoted() {
    let mut promotions = 0;
    for seed in 0..10 {
        let cfg = SimConfig { days: 90, shock_day: None, seed, ..Default::default() };
        let same = (local_edit("same"), Quality { rate: cfg.base_rate, guardrail_rate: 0.001, born_day: 0 });
        let (_, report) = run_fixed(&cfg, vec![same], config(&cfg)).await;
        promotions += report.promotions;
    }
    // The promotion test's error rate is 5%; allow for chance, not a pattern.
    assert!(promotions <= 1, "{promotions} false promotions in 10 A/A runs");
}

#[tokio::test]
async fn a_harmful_challenger_is_rolled_back() {
    let cfg = SimConfig { days: 60, shock_day: None, ..Default::default() };
    let toxic = (local_edit("toxic"), Quality { rate: 0.2, guardrail_rate: 0.05, born_day: 0 });
    let (engine, report) = run_fixed(&cfg, vec![toxic], config(&cfg)).await;
    assert_eq!(report.rollbacks, 1, "{report:?}");
    assert_eq!(report.promotions, 0, "a higher conversion rate never buys a guardrail breach");
    let state = engine.state("p").await.unwrap();
    let rec = state.strategies.values().find(|r| r.strategy.name == "toxic").unwrap();
    assert!(matches!(rec.stage, Stage::RolledBack { .. }));
}

#[tokio::test]
async fn a_drop_in_the_incumbent_is_detected_and_diagnosed() {
    let cfg = SimConfig { days: 110, shock_day: Some(50), shock_factor: 0.35, ..Default::default() };
    let (engine, report) = run_fixed(&cfg, vec![], config(&cfg)).await;
    assert!(report.drift_alarms >= 1, "{report:?}");
    let events = engine.events("p").await.unwrap();
    let (at, diagnosis) = events
        .iter()
        .find_map(|e| match &e.kind {
            StrategyEventKind::DriftDetected { diagnosis, .. } => Some((e.at, diagnosis.clone())),
            _ => None,
        })
        .unwrap();
    let shock = Utc.with_ymd_and_hms(2026, 1, 5, 0, 0, 0).unwrap() + Duration::days(50);
    assert!(at > shock && at < shock + Duration::days(40), "alarm at {at}");
    let success = diagnosis.iter().find(|s| s.event == cfg.success_event).expect("success shift");
    assert!(success.after < success.before);
    assert!(events.iter().any(|e| matches!(e.kind, StrategyEventKind::ProposalsRequested { .. })));
}

#[tokio::test]
async fn it_calibrates_its_maturity_window_from_observed_lags() {
    let cfg = SimConfig { days: 60, shock_day: None, mean_lag_hours: 12.0, ..Default::default() };
    let (engine, _) = run_fixed(&cfg, vec![], config(&cfg)).await;
    let state = engine.state("p").await.unwrap();
    assert!(state.maturity_hours < 168.0, "window stayed at {}", state.maturity_hours);
    let events = engine.events("p").await.unwrap();
    let steps: Vec<f64> = events
        .iter()
        .filter_map(|e| match &e.kind {
            StrategyEventKind::Calibrated { param, old, new, .. } if param == "maturity_hours" => {
                Some(old.as_f64().unwrap() / new.as_f64().unwrap())
            }
            _ => None,
        })
        .collect();
    assert!(!steps.is_empty());
    assert!(steps.iter().all(|r| *r <= 1.5 + 1e-9), "every step is bounded: {steps:?}");
}

#[tokio::test]
async fn simulation_recovers_from_a_market_shift_and_stays_safe() {
    let mut lifts = Vec::new();
    for seed in 1..=4 {
        let cfg = SimConfig { days: 200, seed, ..Default::default() };
        let clock = clock();
        let engine = StrategyEngine::new(Arc::new(InMemoryStore::new())).clock(clock.clone()).seeded_ids(seed);
        let market = Arc::new(Market::default());
        sim::seed_baseline(&market, "baseline", &cfg);
        engine.create_program("p", config(&cfg), baseline()).await.unwrap();
        let proposer = SimProposer::new(market.clone(), &cfg);
        let report = sim::run(&engine, &clock, "p", &cfg, market.clone(), &proposer).await.unwrap();
        eprintln!("seed {seed}: lift {:+.1}% {report:?}", report.lift() * 100.0);
        assert!(report.budget_respected);
        // Harmful strategies are never promoted, and each gets a capped
        // canary at most: roughly the handful of complaints needed to prove
        // harm.
        let state = engine.state("p").await.unwrap();
        for r in state.strategies.values() {
            let q = market.get(&r.strategy.name).unwrap();
            if q.guardrail_rate > cfg.guardrail_ok {
                assert!(!matches!(r.stage, Stage::Incumbent), "harmful strategy promoted");
                let n = state.assignments.values().filter(|a| a.strategy == r.strategy.id).count();
                assert!(n <= 500, "{} reached {n} contacts", r.strategy.name);
            }
        }
        assert!(engine.status("p").await.unwrap().holdout.lift.is_some());
        lifts.push(report.lift());
    }
    let mean = lifts.iter().sum::<f64>() / lifts.len() as f64;
    assert!(lifts.iter().all(|l| *l > 0.0), "never worse than standing still: {lifts:?}");
    assert!(mean > 0.15, "mean lift {mean:.3}: {lifts:?}");
}

#[tokio::test]
async fn health_check_finds_the_drop_and_judges_feasibility() {
    let mut rng = cairn_strategy::rng::Rng::seeded(4);
    let start = NaiveDate::from_ymd_opt(2026, 6, 1).unwrap();
    let records: Vec<DailyRecord> = (0..120)
        .map(|d| {
            let (p, bounce) = if d < 70 { (0.10, 0.01) } else { (0.05, 0.08) };
            let sends = 80u64;
            let count =
                |rng: &mut cairn_strategy::rng::Rng, p: f64| (0..sends).filter(|_| rng.uniform() < p).count() as u64;
            DailyRecord {
                date: start + Duration::days(d),
                sends,
                events: BTreeMap::from([
                    ("reply".into(), count(&mut rng, p)),
                    ("bounce".into(), count(&mut rng, bounce)),
                ]),
            }
        })
        .collect();
    let report = health_check(&records, "reply", 0.02);
    let change = report.changes.last().expect("change found");
    let began = (change.began - start).num_days();
    assert!((63..=77).contains(&began), "began day {began}");
    assert!(change.detected >= change.began);
    assert_eq!(change.shifts[0].event, "bounce", "the biggest shift is deliverability");
    assert!(report.power.weekly_volume > 500.0);
    assert!(report.summary.contains("flagged"));
}

#[tokio::test]
async fn jsonl_store_round_trips_and_detects_tampering() {
    let dir = std::env::temp_dir().join(format!("cairn-strategy-{}", uuid::Uuid::now_v7()));
    let clock = clock();
    let engine = StrategyEngine::new(Arc::new(JsonlStore::new(&dir).unwrap())).clock(clock.clone());
    engine
        .create_program("p", ProgramConfig { success_event: "won".into(), ..Default::default() }, baseline())
        .await
        .unwrap();
    let a = engine.assign("p", "s1").await.unwrap();
    engine.report("p", &a.assignment_id, "won", None).await.unwrap();
    assert!(!engine.report("p", &a.assignment_id, "won", None).await.unwrap(), "repeat is a no-op");

    let fresh = StrategyEngine::new(Arc::new(JsonlStore::new(&dir).unwrap()));
    assert_eq!(fresh.state("p").await.unwrap(), engine.state("p").await.unwrap());

    let path = dir.join("p.jsonl");
    let text = std::fs::read_to_string(&path).unwrap().replace("\"won\"", "\"lost\"");
    std::fs::write(&path, text).unwrap();
    let tampered = StrategyEngine::new(Arc::new(JsonlStore::new(&dir).unwrap()));
    assert!(matches!(tampered.state("p").await, Err(EngineError::Tampered { .. })));
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn model_proposer_drafts_go_through_screening() {
    let clock = clock();
    let engine = engine(clock.clone());
    engine
        .create_program("p", ProgramConfig { success_event: "won".into(), ..Default::default() }, baseline())
        .await
        .unwrap();
    let reply = r#"Here you go: {"name": "audit offer", "descriptor": {"segment": "saas", "channel": "email"},
        "playbook": ["Offer a free 20-minute pipeline audit", "Follow up after 3 days with one stat"],
        "hypothesis": "A concrete, low-effort offer lifts replies", "predicted_lift": 0.2}"#;
    let model = Arc::new(ScriptedModel::new(vec![text_response(reply)]));
    let proposer = ModelProposer::new(model.clone());
    let out = engine.generate("p", &proposer, 1).await.unwrap();
    assert_eq!(out.len(), 1);
    let state = engine.state("p").await.unwrap();
    let rec = &state.strategies[&out[0].id];
    assert_eq!(rec.strategy.proposer.as_deref(), Some("model:scripted:test"));
    assert!(rec.strategy.operator.is_some());
    let prompt = &model.requests()[0];
    assert!(prompt.messages[0].text().contains("already_tried"));
}

#[cfg(feature = "server")]
#[tokio::test]
async fn server_speaks_mcp_http_and_openfeature_with_roles() {
    use cairn_strategy::server::{Access, Role, StrategyServer};
    use serde_json::{Value, json};

    let engine = Arc::new(StrategyEngine::new(Arc::new(InMemoryStore::new())));
    let access =
        Access::open().token("agent-token", "sdr-agent", Role::Agent).token("admin-token", "maria", Role::Admin);
    let app = StrategyServer::new(engine.clone()).access(access).router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum_serve(listener, app).await });
    let http = reqwest::Client::new();
    let post =
        |path: &str, token: &str, body: Value| http.post(format!("{base}{path}")).bearer_auth(token).json(&body).send();

    let created = post(
        "/v1/programs",
        "admin-token",
        json!({
            "name": "outbound", "config": { "success_event": "meeting_booked" },
            "baseline": { "name": "current", "playbook": ["Lead with the compliance pain"] }
        }),
    )
    .await
    .unwrap();
    assert_eq!(created.status(), 200);
    let denied =
        post("/v1/programs", "agent-token", json!({ "name": "x", "baseline": { "name": "b", "playbook": ["a"] } }))
            .await
            .unwrap();
    assert_eq!(denied.status(), 403);

    let rpc = |method: &str, params: Value| json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params });
    let init: Value = post("/mcp", "agent-token", rpc("initialize", json!({}))).await.unwrap().json().await.unwrap();
    assert_eq!(init["result"]["serverInfo"]["name"], "cairn-strategy");
    let list: Value = post("/mcp", "agent-token", rpc("tools/list", json!({}))).await.unwrap().json().await.unwrap();
    let names: Vec<&str> =
        list["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"strategy_assign") && !names.iter().any(|n| n.contains("approve")));

    let call: Value = post(
        "/mcp",
        "agent-token",
        rpc(
            "tools/call",
            json!({
                "name": "strategy_assign", "arguments": { "program": "outbound", "subject": "acme" }
            }),
        ),
    )
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    let assignment = &call["result"]["structuredContent"];
    let id = assignment["assignment_id"].as_str().unwrap().to_owned();
    let reported: Value = post(
        "/mcp",
        "agent-token",
        rpc(
            "tools/call",
            json!({
                "name": "strategy_report_outcome",
                "arguments": { "program": "outbound", "assignment_id": id, "event": "meeting_booked" }
            }),
        ),
    )
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    assert_eq!(reported["result"]["structuredContent"]["recorded"], true);

    let flag: Value =
        post("/ofrep/v1/evaluate/flags/outbound", "agent-token", json!({ "context": { "targetingKey": "acme" } }))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    assert_eq!(flag["metadata"]["assignmentId"], assignment["assignment_id"], "same subject, same assignment");

    let proposal: Value = post(
        "/v1/programs/outbound/strategies",
        "agent-token",
        json!({
            "name": "new segment", "descriptor": { "segment": "healthtech" }, "playbook": ["Lead with HIPAA audit prep"]
        }),
    )
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    let pid = proposal["id"].as_str().unwrap();
    let agent_approve =
        post(&format!("/v1/programs/outbound/strategies/{pid}/approve"), "agent-token", json!({})).await.unwrap();
    assert_eq!(agent_approve.status(), 403, "agents cannot approve");
    let ok = post(&format!("/v1/programs/outbound/strategies/{pid}/approve"), "admin-token", json!({})).await.unwrap();
    assert_eq!(ok.status(), 200);
    let state = engine.state("outbound").await.unwrap();
    assert_eq!(state.strategies[pid].stage, Stage::Approved);
}

#[cfg(feature = "server")]
async fn axum_serve(listener: tokio::net::TcpListener, app: axum::Router) {
    axum::serve(listener, app).await.unwrap();
}
