//! # cairn-strategy
//!
//! A self-calibrating strategy engine for business agents. It notices when an
//! agent's playbook stops working, works out where it broke, brings in new
//! strategies (from people or models), and adopts the ones that win on real
//! outcomes, without letting a model make the call.
//!
//! Two planes:
//!
//! - **Untrusted, generative.** [`Proposer`]s (any Cairn model, or a human)
//!   draft strategies. Drafts are screened by deterministic rules and a
//!   novelty gate, and scope changes always wait for a human.
//! - **Trusted, deterministic.** [`StrategyEngine`] allocates traffic by
//!   Thompson sampling under a conservative loss budget, watches the
//!   incumbent with an e-detector whose false-alarm rate is set up front,
//!   rolls back challengers that breach guardrails, promotes only on
//!   anytime-valid evidence, and keeps a frozen holdout to measure real lift.
//!
//! The engine calibrates itself (maturity window from observed conversion
//! lags, priors by empirical Bayes, operator mix and proposer trust from
//! which past proposals actually won), always from matured outcomes, in
//! bounded steps, and journals every change. Every decision is an event in a
//! hash-chained log ([`verify_chain`]), and state is only a fold over it.
//!
//! Agents in any framework use it over MCP or HTTP (see `server`), or in
//! process:
//!
//! ```ignore
//! let engine = StrategyEngine::new(Arc::new(JsonlStore::new("strategy")?));
//! engine.create_program("outbound", config, Strategy::new("current", playbook)).await?;
//! let a = engine.assign("outbound", "prospect-42").await?;
//! // ... the agent follows a.strategy.playbook ...
//! engine.report("outbound", &a.assignment_id, "meeting_booked", None).await?;
//! engine.tick("outbound").await?; // on a schedule
//! ```

pub mod archive;
pub mod engine;
pub mod event;
pub mod health;
pub mod propose;
pub mod rng;
pub mod screen;
#[cfg(feature = "server")]
pub mod server;
pub mod sim;
pub mod state;
pub mod stats;
pub mod store;
pub mod types;

pub use engine::{
    ArmStatus, Clock, EngineError, HoldoutStatus, ManualClock, ProgramStatus, Proposal, StrategyEngine, SystemClock,
};
pub use event::{Shift, StrategyEvent, StrategyEventKind, verify_chain};
pub use health::{DailyRecord, HealthReport, health_check};
pub use propose::{ArchiveEntry, ModelProposer, ProposalRequest, Proposer, ProposerError};
pub use screen::{Rules, Screen, Similarity, TokenJaccard};
pub use state::ProgramState;
pub use store::{InMemoryStore, JsonlStore, StoreError, StrategyStore};
pub use types::{Assignment, Guardrail, Operator, ProgramConfig, Stage, Strategy, StrategyId};
