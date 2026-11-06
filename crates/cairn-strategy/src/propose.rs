//! The untrusted generative plane. A proposer only ever produces drafts:
//! every draft is screened, usually approved by a human, canaried under a
//! loss budget and judged on real outcomes before it can matter.

use std::sync::Arc;

use async_trait::async_trait;
use cairn_runtime::{Content, Message, Model, ModelRequest};
use serde::{Deserialize, Serialize};

use crate::event::Shift;
use crate::types::{Operator, Stage, Strategy};

/// One archived strategy as a proposer sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub strategy: Strategy,
    pub stage: Stage,
    /// Matured success rate, when it has had traffic.
    pub rate: Option<f64>,
    pub outcomes: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposalRequest {
    pub goal: String,
    pub operator: Operator,
    pub parents: Vec<ArchiveEntry>,
    /// Everything tried so far, failures included, so proposers do not repeat
    /// them.
    pub archive: Vec<ArchiveEntry>,
    pub rules: Vec<String>,
    /// Outcome shifts behind the latest drift alarm, if any. Treat them as
    /// hypotheses to test, not facts.
    pub diagnosis: Vec<Shift>,
    pub scope_axes: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ProposerError(pub String);

#[async_trait]
pub trait Proposer: Send + Sync {
    /// Recorded on every draft, so proposers build a track record.
    fn name(&self) -> String;
    async fn propose(&self, request: &ProposalRequest) -> Result<Vec<Strategy>, ProposerError>;
}

/// Proposes strategies with any Cairn [`Model`].
pub struct ModelProposer {
    model: Arc<dyn Model>,
    max_tokens: u32,
}

impl ModelProposer {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self { model, max_tokens: 2000 }
    }
}

const SYSTEM: &str = "You design business strategies (playbooks) for an AI agent. \
You propose; statistics and humans decide. Never claim a result you cannot know. \
Reply with one JSON object only: {\"name\": str, \"descriptor\": {axis: value}, \
\"playbook\": [str], \"hypothesis\": str, \"predicted_lift\": number}. \
predicted_lift is your honest estimate of relative lift over the incumbent (0.1 = +10%); \
it is calibrated against real outcomes, so overconfidence only lowers your weight.";

fn operator_instruction(op: Operator) -> &'static str {
    match op {
        Operator::Local => {
            "Make one or two small, itemized edits to the parent's playbook. Keep its descriptor unchanged."
        }
        Operator::AxisShift => {
            "Change exactly one descriptor axis of the parent (e.g. a new segment, channel, offer or angle) \
             and adapt the playbook to it."
        }
        Operator::Crossover => "Combine the strongest elements of the two parents into one coherent playbook.",
        Operator::Fresh => {
            "Ignore the parents. Design a genuinely different strategy aimed at a descriptor cell nobody has \
             tried yet."
        }
    }
}

#[async_trait]
impl Proposer for ModelProposer {
    fn name(&self) -> String {
        format!("model:{}", self.model.id())
    }

    async fn propose(&self, req: &ProposalRequest) -> Result<Vec<Strategy>, ProposerError> {
        let brief = serde_json::json!({
            "goal": req.goal,
            "task": operator_instruction(req.operator),
            "parents": req.parents,
            "already_tried": req.archive.iter().map(|e| serde_json::json!({
                "name": e.strategy.name,
                "descriptor": e.strategy.descriptor,
                "stage": e.stage,
                "rate": e.rate,
                "outcomes": e.outcomes,
            })).collect::<Vec<_>>(),
            "rules": req.rules,
            "recent_shifts": req.diagnosis,
            "scope_axes_need_human_approval": req.scope_axes,
        });
        let request = ModelRequest {
            system: Some(SYSTEM.into()),
            messages: vec![Message::user(serde_json::to_string_pretty(&brief).unwrap_or_default())],
            max_tokens: self.max_tokens,
            ..Default::default()
        };
        let response = self.model.complete(&request).await.map_err(|e| ProposerError(e.to_string()))?;
        let text: String = response
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let start = text.find('{').ok_or_else(|| ProposerError("no JSON object in reply".into()))?;
        let mut de = serde_json::Deserializer::from_str(&text[start..]).into_iter::<Strategy>();
        let strategy =
            de.next().ok_or_else(|| ProposerError("empty reply".into()))?.map_err(|e| ProposerError(e.to_string()))?;
        Ok(vec![strategy])
    }
}
