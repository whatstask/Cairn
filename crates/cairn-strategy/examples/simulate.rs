//! Run the strategy engine against a synthetic market and print what it did.
//!
//! ```text
//! cargo run --release -p cairn-ai-strategy --example simulate -- [seed] [days] [contacts_per_day]
//! ```
//!
//! The market has a hidden truth: each strategy's real conversion and
//! complaint rates. Halfway through, the market shifts and existing
//! playbooks lose half their effect. The run reports the gain over keeping
//! the original strategy, and every strategy's fate.

use std::sync::Arc;

use cairn_strategy::sim::{self, Market, SimConfig, SimProposer};
use cairn_strategy::{Guardrail, InMemoryStore, ManualClock, ProgramConfig, Strategy, StrategyEngine};

#[tokio::main]
async fn main() {
    let args: Vec<u64> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let cfg = SimConfig {
        seed: args.first().copied().unwrap_or(1),
        days: args.get(1).copied().unwrap_or(200) as u32,
        contacts_per_day: args.get(2).copied().unwrap_or(100) as u32,
        ..Default::default()
    };
    let clock = Arc::new(ManualClock::new(sim::start()));
    let engine = StrategyEngine::new(Arc::new(InMemoryStore::new())).clock(clock.clone()).seeded_ids(cfg.seed);
    let market = Arc::new(Market::default());
    sim::seed_baseline(&market, "baseline", &cfg);
    let config = ProgramConfig {
        goal: "book meetings with B2B buyers".into(),
        success_event: cfg.success_event.clone(),
        guardrails: vec![Guardrail { event: cfg.guardrail_event.clone(), max_rate: 0.005 }],
        ..Default::default()
    };
    let mut baseline = Strategy::new("baseline", vec!["Open with the customer's compliance pain".into()]);
    baseline.descriptor = sim::baseline_descriptor();
    let program = std::env::var("PROGRAM").unwrap_or("demo".into());
    engine.create_program(&program, config, baseline).await.unwrap();
    let proposer = SimProposer::new(market.clone(), &cfg);
    let report = sim::run(&engine, &clock, &program, &cfg, market.clone(), &proposer).await.unwrap();

    println!(
        "{} days, {} contacts: {} successes, {:+.1}% vs keeping the original strategy",
        cfg.days,
        report.contacts,
        report.successes,
        report.lift() * 100.0
    );
    println!(
        "promotions {} · rollbacks {} · drift alarms {} · proposals {} · budget respected {}",
        report.promotions, report.rollbacks, report.drift_alarms, report.proposals, report.budget_respected
    );
    let state = engine.state(&program).await.unwrap();
    let start = state.created_at;
    println!("\n{:<9} {:>5} {:>6} {:>5} {:>8} {:>8}  stage", "strategy", "born", "rate", "harm", "contacts", "matured");
    for r in state.strategies.values() {
        let q = market.get(&r.strategy.name).unwrap();
        let n = state.assignments.values().filter(|a| a.strategy == r.strategy.id).count();
        let since = r.live_at.unwrap_or(start);
        let c = state.matured_counts(&r.strategy.id, since, engine.now(), false);
        let live = r.live_at.map(|t| (t - start).num_days().to_string()).unwrap_or("-".into());
        println!(
            "{:<9} {:>5} {:>6.3} {:>5} {:>8} {:>8.0}  {:?} (day {})",
            r.strategy.name,
            live,
            q.rate,
            if q.guardrail_rate > cfg.guardrail_ok { "yes" } else { "" },
            n,
            c.trials,
            r.stage,
            (r.stage_at - start).num_days()
        );
    }
    let status = engine.status(&program).await.unwrap();
    println!(
        "\nholdout lift {:?} · maturity {:.0}h · prior {:.3} (strength {:.1}) · prediction trust {:.2}",
        status.holdout.lift.map(|l| format!("{:+.1}%", l * 100.0)),
        status.maturity_hours,
        status.prior.mean(),
        status.prior.strength(),
        status.prediction_trust
    );
    println!("event log: {} events, hash chain verified: {}", status.events, engine.events(&program).await.is_ok());
}
