//! # cairn-sandbox
//!
//! Isolated code execution for Cairn agents. Each run gets its own sandbox,
//! a hardened container with a private workspace, and the model works in it
//! through four tools: `bash`, `read_file`, `write_file` and `edit_file`.
//!
//! ```ignore
//! use cairn_sandbox::{DockerProvider, SandboxOptions, SandboxTools};
//!
//! let sandboxes = SandboxTools::new(DockerProvider::new("python:3.12-slim"), SandboxOptions::default());
//! let agent = Agent::builder("analyst").model(model).tools(sandboxes.tools()).build()?;
//!
//! // Put inputs in place, run, collect results, clean up.
//! sandboxes.sandbox(&run_id).await?.write_file("data.csv", &csv).await?;
//! agent.run_with_id(&run_id, "Summarise data.csv by month").await?;
//! let report = sandboxes.sandbox(&run_id).await?.read_file("report.md", 1 << 20).await?;
//! sandboxes.close(&run_id).await?;
//! ```
//!
//! ## Isolation
//!
//! [`DockerProvider`] runs each sandbox as a container with no network, a
//! read-only root filesystem, every Linux capability dropped,
//! `no-new-privileges`, memory, CPU and process limits, and size-capped
//! in-memory writable mounts. Commands run as an unprivileged user. A
//! timeout kills every process the command started. Each container removes
//! itself when its lifetime cap is reached, so a crash leaves nothing behind
//! for long. For untrusted code in production, add gVisor
//! ([`DockerProvider::runtime`]`("runsc")`).
//!
//! The `bash` tool's [effect](cairn_runtime::Effect) follows the network:
//! `Write` when commands cannot reach any network, `Irreversible` (approval
//! required under the default policy) when they can.
//!
//! ## Durability
//!
//! Sandboxes are keyed by run id. A run resumed after a crash reattaches to
//! its sandbox if it is still alive; if it has expired, the model is told
//! its files were lost and the next call starts a fresh one.
//!
//! Other backends (microVMs, remote sandbox services) plug in by
//! implementing [`SandboxProvider`] and [`Sandbox`].

mod docker;
mod sandbox;
mod tools;

pub use docker::{DockerProvider, DockerSandbox, Network};
pub use sandbox::{ExecOutput, FileContents, Sandbox, SandboxError, SandboxProvider, WORKSPACE};
pub use tools::{SandboxOptions, SandboxTool, SandboxTools};
