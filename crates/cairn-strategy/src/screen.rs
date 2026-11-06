//! Deterministic screening of proposals. Nothing here calls a model: a
//! proposal that breaks a rule, or duplicates something already tried
//! (including failures), never reaches a human or traffic.

use std::collections::{BTreeMap, BTreeSet};

use crate::state::ProgramState;
use crate::types::{Operator, Strategy};

/// Hard rules on what a strategy may say and where it may go. Only humans
/// edit them; proposers see them as constraints.
pub trait Screen: Send + Sync {
    /// Violations; empty means the strategy passes.
    fn check(&self, strategy: &Strategy) -> Vec<String>;

    /// Plain-language rules for proposers.
    fn describe(&self) -> Vec<String> {
        Vec::new()
    }
}

/// A rule set covering the common cases.
#[derive(Debug, Clone, Default)]
pub struct Rules {
    /// Allowed values per descriptor axis; an axis not listed is open.
    pub allowed: BTreeMap<String, Vec<String>>,
    /// Axes every strategy must set.
    pub required_axes: Vec<String>,
    /// Case-insensitive phrases that must not appear in a playbook
    /// (e.g. "guaranteed results", "act now").
    pub banned_terms: Vec<String>,
    /// Upper bound on playbook items.
    pub max_items: Option<usize>,
}

impl Screen for Rules {
    fn check(&self, s: &Strategy) -> Vec<String> {
        let mut out = Vec::new();
        if s.playbook.is_empty() {
            out.push("playbook is empty".into());
        }
        if let Some(max) = self.max_items {
            if s.playbook.len() > max {
                out.push(format!("playbook has {} items, limit is {max}", s.playbook.len()));
            }
        }
        for axis in &self.required_axes {
            if !s.descriptor.contains_key(axis) {
                out.push(format!("descriptor is missing `{axis}`"));
            }
        }
        for (axis, value) in &s.descriptor {
            if let Some(allowed) = self.allowed.get(axis) {
                if !allowed.iter().any(|a| a.eq_ignore_ascii_case(value)) {
                    out.push(format!("`{axis}` = `{value}` is not allowed"));
                }
            }
        }
        let text = s.playbook.join("\n").to_lowercase();
        for term in &self.banned_terms {
            if text.contains(&term.to_lowercase()) {
                out.push(format!("uses banned phrase `{term}`"));
            }
        }
        out
    }

    fn describe(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (axis, values) in &self.allowed {
            out.push(format!("`{axis}` must be one of: {}", values.join(", ")));
        }
        if !self.required_axes.is_empty() {
            out.push(format!("set these descriptor axes: {}", self.required_axes.join(", ")));
        }
        if !self.banned_terms.is_empty() {
            out.push(format!("never use: {}", self.banned_terms.join(", ")));
        }
        if let Some(max) = self.max_items {
            out.push(format!("at most {max} playbook items"));
        }
        out
    }
}

/// Similarity in [0, 1] between two strategies, used for the novelty gate.
/// Plug in an embedding model for semantic similarity; the default compares
/// word sets.
pub trait Similarity: Send + Sync {
    fn similarity(&self, a: &Strategy, b: &Strategy) -> f64;
}

/// Jaccard similarity of lower-cased word sets over descriptor and playbook.
pub struct TokenJaccard;

impl Similarity for TokenJaccard {
    fn similarity(&self, a: &Strategy, b: &Strategy) -> f64 {
        let words = |s: &Strategy| -> BTreeSet<String> {
            s.descriptor
                .values()
                .chain(s.playbook.iter())
                .flat_map(|t| t.split(|c: char| !c.is_alphanumeric()))
                .filter(|w| !w.is_empty())
                .map(|w| w.to_lowercase())
                .collect()
        };
        let (x, y) = (words(a), words(b));
        let union = x.union(&y).count();
        if union == 0 { 1.0 } else { x.intersection(&y).count() as f64 / union as f64 }
    }
}

/// Screen a proposal against rules, the archive and approval policy.
/// Returns `(violations, needs_approval)`.
pub fn screen(
    state: &ProgramState,
    strategy: &Strategy,
    rules: &dyn Screen,
    similarity: &dyn Similarity,
) -> (Vec<String>, bool) {
    let mut reasons = rules.check(strategy);
    for p in &strategy.parents {
        if !state.strategies.contains_key(p) {
            reasons.push(format!("unknown parent `{p}`"));
        }
    }
    for (id, rec) in &state.strategies {
        if id == &strategy.id {
            continue;
        }
        let sim = similarity.similarity(strategy, &rec.strategy);
        if sim >= state.config.novelty_threshold {
            reasons.push(format!("duplicates `{}` ({id}, similarity {sim:.2})", rec.strategy.name));
            break;
        }
    }
    (reasons, needs_approval(state, strategy))
}

/// Scope changes (any of the configured scope axes differs from every
/// parent, or a fresh strategy) always need a human. Local edits need one
/// unless the program auto-approves them.
pub fn needs_approval(state: &ProgramState, strategy: &Strategy) -> bool {
    let parents: Vec<&Strategy> =
        strategy.parents.iter().filter_map(|p| state.strategies.get(p)).map(|r| &r.strategy).collect();
    let scope_change = parents.is_empty()
        || state.config.scope_axes.iter().any(|axis| {
            parents.iter().all(|p| p.descriptor.get(axis) != strategy.descriptor.get(axis))
                && (strategy.descriptor.contains_key(axis) || parents.iter().any(|p| p.descriptor.contains_key(axis)))
        });
    if scope_change {
        return true;
    }
    !(state.config.auto_approve_local && strategy.operator.unwrap_or(Operator::Local) == Operator::Local)
}
