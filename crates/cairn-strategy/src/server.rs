//! HTTP surface for agents in any framework.
//!
//! - `POST /mcp`: a Model Context Protocol server (Streamable HTTP, JSON
//!   responses). Microsoft Agent Framework, AWS Bedrock AgentCore Gateway,
//!   Azure AI Foundry, Google ADK, the OpenAI Agents SDK, LangGraph, CrewAI
//!   and the Claude Agent SDK all connect to remote MCP servers.
//! - `/v1/...`: a JSON API (described at `/openapi.json`) for code paths that
//!   should assign deterministically rather than leave the call to a model;
//!   that keeps logged propensities trustworthy.
//! - `POST /ofrep/v1/evaluate/flags/{program}`: OpenFeature Remote
//!   Evaluation, so feature-flag SDKs can read assignments.
//!
//! Approvals, rejections, pausing and configuration need an admin token and
//! are deliberately not exposed as MCP tools: models propose, people decide.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::engine::{EngineError, StrategyEngine};
use crate::health::{DailyRecord, health_check};
use crate::types::{ProgramConfig, Strategy};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// Agents: assign, report outcomes, propose, read status.
    Agent,
    /// People: everything, including approvals and the kill switch.
    Admin,
}

/// Bearer tokens and who they belong to. With no tokens configured the
/// server is open (for local development only).
#[derive(Debug, Clone, Default)]
pub struct Access {
    tokens: HashMap<String, (String, Role)>,
}

impl Access {
    pub fn open() -> Self {
        Self::default()
    }

    pub fn token(mut self, token: impl Into<String>, principal: impl Into<String>, role: Role) -> Self {
        self.tokens.insert(token.into(), (principal.into(), role));
        self
    }

    fn check(&self, headers: &HeaderMap, need: Role) -> Result<String, ApiError> {
        if self.tokens.is_empty() {
            return Ok("anonymous".into());
        }
        let token = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .ok_or(ApiError(StatusCode::UNAUTHORIZED, "missing bearer token".into()))?;
        match self.tokens.get(token) {
            Some((who, role)) if *role >= need => Ok(who.clone()),
            Some(_) => Err(ApiError(StatusCode::FORBIDDEN, "this token cannot do that".into())),
            None => Err(ApiError(StatusCode::UNAUTHORIZED, "unknown token".into())),
        }
    }
}

#[derive(Clone)]
struct App {
    engine: Arc<StrategyEngine>,
    access: Access,
}

pub struct StrategyServer {
    app: App,
}

impl StrategyServer {
    pub fn new(engine: Arc<StrategyEngine>) -> Self {
        Self { app: App { engine, access: Access::open() } }
    }

    pub fn access(mut self, access: Access) -> Self {
        self.app.access = access;
        self
    }

    pub fn router(self) -> Router {
        Router::new()
            .route("/mcp", post(mcp))
            .route("/openapi.json", get(openapi))
            .route("/healthz", get(|| async { "ok" }))
            .route("/v1/programs", get(list_programs).post(create_program))
            .route("/v1/programs/{program}/status", get(program_status))
            .route("/v1/programs/{program}/assign", post(assign))
            .route("/v1/programs/{program}/outcomes", post(report))
            .route("/v1/programs/{program}/strategies", post(propose))
            .route("/v1/programs/{program}/strategies/{id}/approve", post(approve))
            .route("/v1/programs/{program}/strategies/{id}/reject", post(reject))
            .route("/v1/programs/{program}/pause", post(pause))
            .route("/v1/programs/{program}/resume", post(resume))
            .route("/v1/programs/{program}/config", post(update_config))
            .route("/v1/programs/{program}/tick", post(tick))
            .route("/v1/programs/{program}/events", get(events))
            .route("/v1/health-check", post(health))
            .route("/ofrep/v1/evaluate/flags/{program}", post(ofrep))
            .with_state(self.app)
    }

    pub async fn serve(self, addr: &str) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!(%addr, "strategy engine listening");
        axum::serve(listener, self.router()).await
    }
}

struct ApiError(StatusCode, String);

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        let code = match &e {
            EngineError::NotFound(_) => StatusCode::NOT_FOUND,
            EngineError::Exists(_) => StatusCode::CONFLICT,
            EngineError::Invalid(_) => StatusCode::UNPROCESSABLE_ENTITY,
            EngineError::Proposer(_) => StatusCode::BAD_GATEWAY,
            EngineError::Store(_) | EngineError::Tampered { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        };
        ApiError(code, e.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

type ApiResult = Result<Json<Value>, ApiError>;

fn to_json<T: serde::Serialize>(v: T) -> ApiResult {
    Ok(Json(serde_json::to_value(v).unwrap_or(Value::Null)))
}

#[derive(Deserialize)]
struct CreateBody {
    name: String,
    #[serde(default)]
    config: ProgramConfig,
    baseline: Strategy,
}

async fn create_program(State(app): State<App>, headers: HeaderMap, Json(b): Json<CreateBody>) -> ApiResult {
    app.access.check(&headers, Role::Admin)?;
    let id = app.engine.create_program(&b.name, b.config, b.baseline).await?;
    to_json(json!({ "program": b.name, "baseline": id }))
}

async fn list_programs(State(app): State<App>, headers: HeaderMap) -> ApiResult {
    app.access.check(&headers, Role::Agent)?;
    to_json(app.engine.list_programs().await?)
}

async fn program_status(State(app): State<App>, headers: HeaderMap, Path(program): Path<String>) -> ApiResult {
    app.access.check(&headers, Role::Agent)?;
    to_json(app.engine.status(&program).await?)
}

#[derive(Deserialize)]
struct AssignBody {
    subject: String,
}

async fn assign(
    State(app): State<App>,
    headers: HeaderMap,
    Path(program): Path<String>,
    Json(b): Json<AssignBody>,
) -> ApiResult {
    app.access.check(&headers, Role::Agent)?;
    to_json(app.engine.assign(&program, &b.subject).await?)
}

#[derive(Deserialize)]
struct OutcomeBody {
    assignment_id: String,
    event: String,
    #[serde(default)]
    occurred_at: Option<DateTime<Utc>>,
}

async fn report(
    State(app): State<App>,
    headers: HeaderMap,
    Path(program): Path<String>,
    Json(b): Json<OutcomeBody>,
) -> ApiResult {
    app.access.check(&headers, Role::Agent)?;
    let recorded = app.engine.report(&program, &b.assignment_id, &b.event, b.occurred_at).await?;
    to_json(json!({ "recorded": recorded }))
}

async fn propose(
    State(app): State<App>,
    headers: HeaderMap,
    Path(program): Path<String>,
    Json(mut s): Json<Strategy>,
) -> ApiResult {
    let who = app.access.check(&headers, Role::Agent)?;
    if s.proposer.is_none() {
        s.proposer = Some(who);
    }
    to_json(app.engine.propose(&program, s).await?)
}

#[derive(Deserialize, Default)]
struct DecisionBody {
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    reason: Option<String>,
}

async fn approve(
    State(app): State<App>,
    headers: HeaderMap,
    Path((program, id)): Path<(String, String)>,
    body: Option<Json<DecisionBody>>,
) -> ApiResult {
    let who = app.access.check(&headers, Role::Admin)?;
    let note = body.and_then(|b| b.0.note);
    app.engine.approve(&program, &id, &who, note).await?;
    to_json(json!({ "approved": id, "by": who }))
}

async fn reject(
    State(app): State<App>,
    headers: HeaderMap,
    Path((program, id)): Path<(String, String)>,
    body: Option<Json<DecisionBody>>,
) -> ApiResult {
    let who = app.access.check(&headers, Role::Admin)?;
    let reason = body.and_then(|b| b.0.reason).unwrap_or_else(|| "rejected by reviewer".into());
    app.engine.reject(&program, &id, &who, &reason).await?;
    to_json(json!({ "rejected": id, "by": who }))
}

async fn pause(
    State(app): State<App>,
    headers: HeaderMap,
    Path(program): Path<String>,
    body: Option<Json<DecisionBody>>,
) -> ApiResult {
    let who = app.access.check(&headers, Role::Admin)?;
    let reason = body.and_then(|b| b.0.reason).unwrap_or_else(|| "paused by operator".into());
    app.engine.pause(&program, &who, &reason).await?;
    to_json(json!({ "paused": true }))
}

async fn resume(State(app): State<App>, headers: HeaderMap, Path(program): Path<String>) -> ApiResult {
    let who = app.access.check(&headers, Role::Admin)?;
    app.engine.resume(&program, &who).await?;
    to_json(json!({ "paused": false }))
}

async fn update_config(
    State(app): State<App>,
    headers: HeaderMap,
    Path(program): Path<String>,
    Json(config): Json<ProgramConfig>,
) -> ApiResult {
    let who = app.access.check(&headers, Role::Admin)?;
    app.engine.update_config(&program, config, &who).await?;
    to_json(json!({ "updated": true }))
}

async fn tick(State(app): State<App>, headers: HeaderMap, Path(program): Path<String>) -> ApiResult {
    app.access.check(&headers, Role::Admin)?;
    to_json(app.engine.tick(&program).await?)
}

async fn events(State(app): State<App>, headers: HeaderMap, Path(program): Path<String>) -> ApiResult {
    app.access.check(&headers, Role::Admin)?;
    to_json(app.engine.events(&program).await?)
}

#[derive(Deserialize)]
struct HealthBody {
    success_event: String,
    #[serde(default = "default_delta")]
    delta: f64,
    records: Vec<DailyRecord>,
}

fn default_delta() -> f64 {
    0.02
}

async fn health(State(app): State<App>, headers: HeaderMap, Json(b): Json<HealthBody>) -> ApiResult {
    app.access.check(&headers, Role::Agent)?;
    to_json(health_check(&b.records, &b.success_event, b.delta))
}

/// OpenFeature Remote Evaluation Protocol: the flag key is the program, the
/// targeting key is the subject, and the value is the strategy.
async fn ofrep(
    State(app): State<App>,
    headers: HeaderMap,
    Path(program): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    if let Err(e) = app.access.check(&headers, Role::Agent) {
        return e.into_response();
    }
    let Some(subject) = body.pointer("/context/targetingKey").and_then(Value::as_str) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "key": program, "errorCode": "TARGETING_KEY_MISSING", "errorDetails": "context.targetingKey is required" })),
        )
            .into_response();
    };
    match app.engine.assign(&program, subject).await {
        Ok(a) => Json(json!({
            "key": program,
            "value": { "id": a.strategy_id, "name": a.strategy.name, "playbook": a.strategy.playbook, "descriptor": a.strategy.descriptor },
            "variant": a.strategy_id,
            "reason": if a.halt { "DISABLED" } else { "SPLIT" },
            "metadata": { "assignmentId": a.assignment_id, "propensity": a.propensity, "holdout": a.holdout, "halt": a.halt },
        }))
        .into_response(),
        Err(EngineError::NotFound(_)) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "key": program, "errorCode": "FLAG_NOT_FOUND" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "key": program, "errorCode": "GENERAL", "errorDetails": e.to_string() })),
        )
            .into_response(),
    }
}

// ── MCP ─────────────────────────────────────────────────────────────────

const MCP_PROTOCOL: &str = "2025-06-18";

fn tools() -> Value {
    json!([
        {
            "name": "strategy_assign",
            "description": "Get the strategy (playbook) to use for one subject, such as a prospect or account. Call before acting on the subject, follow the returned playbook, and keep the assignment_id to report outcomes. If halt is true, do not act.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "program": { "type": "string", "description": "Strategy program name" },
                    "subject": { "type": "string", "description": "Stable id of the prospect/account/ticket" }
                },
                "required": ["program", "subject"]
            }
        },
        {
            "name": "strategy_report_outcome",
            "description": "Report an outcome for an assignment, e.g. reply, meeting_booked, bounce, unsubscribe, spam_complaint. Safe to repeat.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "program": { "type": "string" },
                    "assignment_id": { "type": "string" },
                    "event": { "type": "string" },
                    "occurred_at": { "type": "string", "format": "date-time" }
                },
                "required": ["program", "assignment_id", "event"]
            }
        },
        {
            "name": "strategy_status",
            "description": "Live strategies, their matured results with confidence intervals, the holdout lift, pending approvals and calibration state.",
            "inputSchema": {
                "type": "object",
                "properties": { "program": { "type": "string" } },
                "required": ["program"]
            }
        },
        {
            "name": "strategy_propose",
            "description": "Propose a new strategy. It is screened against the rules and everything already tried; changes of segment, channel or geography wait for human approval, and it only gets traffic as a capped canary.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "program": { "type": "string" },
                    "name": { "type": "string" },
                    "descriptor": { "type": "object", "additionalProperties": { "type": "string" } },
                    "playbook": { "type": "array", "items": { "type": "string" } },
                    "hypothesis": { "type": "string" },
                    "parents": { "type": "array", "items": { "type": "string" } },
                    "predicted_lift": { "type": "number" }
                },
                "required": ["program", "name", "playbook"]
            }
        },
        {
            "name": "strategy_health_check",
            "description": "Replay daily history to find when performance changed, which outcome shifted, and whether volume supports live testing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "success_event": { "type": "string" },
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": { "type": "string", "format": "date" },
                                "sends": { "type": "integer" },
                                "events": { "type": "object", "additionalProperties": { "type": "integer" } }
                            },
                            "required": ["date", "sends"]
                        }
                    }
                },
                "required": ["success_event", "records"]
            }
        }
    ])
}

async fn call_tool(app: &App, name: &str, args: &Value, who: &str) -> Result<Value, String> {
    fn ok<T: serde::Serialize>(r: Result<T, EngineError>) -> Result<Value, String> {
        r.map(|v| serde_json::to_value(v).unwrap_or(Value::Null)).map_err(|e| e.to_string())
    }
    let s = |k: &str| args.get(k).and_then(Value::as_str).map(str::to_owned).ok_or(format!("`{k}` is required"));
    match name {
        "strategy_assign" => ok(app.engine.assign(&s("program")?, &s("subject")?).await),
        "strategy_report_outcome" => {
            let at = args.get("occurred_at").and_then(Value::as_str).and_then(|t| t.parse().ok());
            let r = app.engine.report(&s("program")?, &s("assignment_id")?, &s("event")?, at).await;
            ok(r.map(|recorded| json!({ "recorded": recorded })))
        }
        "strategy_status" => ok(app.engine.status(&s("program")?).await),
        "strategy_propose" => {
            let mut strategy: Strategy = serde_json::from_value(args.clone()).map_err(|e| e.to_string())?;
            strategy.proposer.get_or_insert_with(|| who.to_owned());
            ok(app.engine.propose(&s("program")?, strategy).await)
        }
        "strategy_health_check" => {
            let records: Vec<DailyRecord> =
                serde_json::from_value(args.get("records").cloned().unwrap_or_default()).map_err(|e| e.to_string())?;
            ok(Ok(health_check(&records, &s("success_event")?, 0.02)))
        }
        other => Err(format!("unknown tool `{other}`")),
    }
}

async fn mcp(State(app): State<App>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let who = match app.access.check(&headers, Role::Agent) {
        Ok(w) => w,
        Err(e) => return e.into_response(),
    };
    let id = req.get("id").cloned();
    let method = req.get("method").and_then(Value::as_str).unwrap_or_default();
    // Notifications get no response body.
    if id.is_none() {
        return StatusCode::ACCEPTED.into_response();
    }
    let result: Result<Value, (i64, String)> = match method {
        "initialize" => Ok(json!({
            "protocolVersion": MCP_PROTOCOL,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": "cairn-strategy", "version": env!("CARGO_PKG_VERSION") },
            "instructions": "Call strategy_assign before acting on a subject and follow the returned playbook; report outcomes with strategy_report_outcome. Humans approve scope changes; you can only propose."
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tools() })),
        "tools/call" => {
            let name = req.pointer("/params/name").and_then(Value::as_str).unwrap_or_default();
            let args = req.pointer("/params/arguments").cloned().unwrap_or(json!({}));
            match call_tool(&app, name, &args, &who).await {
                Ok(v) => Ok(json!({
                    "content": [{ "type": "text", "text": v.to_string() }],
                    "structuredContent": v,
                    "isError": false
                })),
                Err(msg) => Ok(json!({ "content": [{ "type": "text", "text": msg }], "isError": true })),
            }
        }
        other => Err((-32601, format!("method `{other}` not found"))),
    };
    let body = match result {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err((code, message)) => json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }),
    };
    Json(body).into_response()
}

async fn openapi() -> Json<Value> {
    let path = |summary: &str, body: Option<Value>| {
        let mut op = json!({ "summary": summary, "responses": { "200": { "description": "OK" } } });
        if let Some(schema) = body {
            op["requestBody"] = json!({ "required": true, "content": { "application/json": { "schema": schema } } });
        }
        op
    };
    let program = json!([{ "name": "program", "in": "path", "required": true, "schema": { "type": "string" } }]);
    Json(json!({
        "openapi": "3.1.0",
        "info": { "title": "Cairn Strategy Engine", "version": env!("CARGO_PKG_VERSION") },
        "components": { "securitySchemes": { "bearer": { "type": "http", "scheme": "bearer" } } },
        "security": [{ "bearer": [] }],
        "paths": {
            "/v1/programs/{program}/assign": { "parameters": program, "post": path("Assign a strategy to a subject",
                Some(json!({ "type": "object", "properties": { "subject": { "type": "string" } }, "required": ["subject"] }))) },
            "/v1/programs/{program}/outcomes": { "parameters": program, "post": path("Report an outcome",
                Some(json!({ "type": "object", "properties": {
                    "assignment_id": { "type": "string" }, "event": { "type": "string" },
                    "occurred_at": { "type": "string", "format": "date-time" } }, "required": ["assignment_id", "event"] }))) },
            "/v1/programs/{program}/status": { "parameters": program, "get": path("Program status", None) },
            "/v1/programs/{program}/strategies": { "parameters": program, "post": path("Propose a strategy",
                Some(json!({ "type": "object", "properties": {
                    "name": { "type": "string" }, "playbook": { "type": "array", "items": { "type": "string" } },
                    "descriptor": { "type": "object" }, "hypothesis": { "type": "string" },
                    "parents": { "type": "array", "items": { "type": "string" } }, "predicted_lift": { "type": "number" } },
                    "required": ["name", "playbook"] }))) },
            "/v1/health-check": { "post": path("Replay history for changes and test feasibility",
                Some(json!({ "type": "object", "properties": {
                    "success_event": { "type": "string" }, "records": { "type": "array" } }, "required": ["success_event", "records"] }))) }
        }
    }))
}
