use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;

/// Where relative paths resolve, and the working directory of every command.
pub const WORKSPACE: &str = "/workspace";

#[derive(Debug, Clone, thiserror::Error)]
pub enum SandboxError {
    /// The sandbox no longer exists (it reached its lifetime cap, or was
    /// removed). Its files are gone; the next open creates a fresh one.
    #[error("sandbox `{0}` no longer exists")]
    Gone(String),
    /// The sandbox could not be created or reached.
    #[error("sandbox backend: {0}")]
    Backend(String),
    /// A file operation failed inside the sandbox (missing file, no space…).
    #[error("{0}")]
    File(String),
}

/// The result of one command.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExecOutput {
    /// `None` when the command was killed (timeout or signal).
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// Some output was left out; the start and the end are kept.
    pub truncated: bool,
    /// The command ran past its timeout, and every process it started was killed.
    pub timed_out: bool,
    pub duration_ms: u64,
}

/// A file read from the sandbox.
#[derive(Debug, Clone, PartialEq)]
pub struct FileContents {
    pub bytes: Vec<u8>,
    /// The file's full size; larger than `bytes.len()` when the read was capped.
    pub size: u64,
}

impl FileContents {
    pub fn truncated(&self) -> bool {
        (self.bytes.len() as u64) < self.size
    }
}

/// One isolated workspace. Commands and file operations on the same sandbox
/// see the same files; nothing outside it is reachable unless the backend's
/// network settings allow it.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// Runs `command` with `sh -c` in [`WORKSPACE`]. On timeout every process
    /// the command started is killed; files stay.
    async fn exec(&self, command: &str, timeout: Duration, max_output: usize) -> Result<ExecOutput, SandboxError>;

    /// Writes a file, creating parent directories. Relative paths resolve
    /// against [`WORKSPACE`].
    async fn write_file(&self, path: &str, contents: &[u8]) -> Result<(), SandboxError>;

    /// Reads at most `max_bytes` of a file.
    async fn read_file(&self, path: &str, max_bytes: usize) -> Result<FileContents, SandboxError>;

    /// Destroys the sandbox and its files.
    async fn close(&self) -> Result<(), SandboxError>;
}

/// Creates sandboxes. Each run gets its own, keyed by run id, so a run
/// resumed after a crash reattaches to its workspace when it still exists.
#[async_trait]
pub trait SandboxProvider: Send + Sync {
    /// Opens the sandbox for `key`, reattaching to a live one.
    async fn open(&self, key: &str) -> Result<Arc<dyn Sandbox>, SandboxError>;

    /// Whether commands are cut off from every network. When they are not,
    /// a command can reach the outside world, so the `bash` tool is
    /// `Irreversible` and needs approval under the default policy.
    fn network_isolated(&self) -> bool;
}

/// Resolves a model-supplied path against [`WORKSPACE`].
pub(crate) fn resolve(path: &str) -> Result<String, SandboxError> {
    let path = path.trim();
    if path.is_empty() {
        return Err(SandboxError::File("path is empty".into()));
    }
    if path.contains('\0') {
        return Err(SandboxError::File("path contains a NUL byte".into()));
    }
    Ok(if path.starts_with('/') { path.to_owned() } else { format!("{WORKSPACE}/{path}") })
}

/// Keeps the first and last `cap / 2` bytes of a stream, so both the start
/// of the output and the final error survive truncation.
#[derive(Debug)]
pub(crate) struct Capture {
    head: Vec<u8>,
    tail: VecDeque<u8>,
    half: usize,
    total: usize,
}

impl Capture {
    pub(crate) fn new(cap: usize) -> Self {
        let half = cap.max(2) / 2;
        Self { head: Vec::new(), tail: VecDeque::new(), half, total: 0 }
    }

    pub(crate) fn push(&mut self, mut chunk: &[u8]) {
        self.total += chunk.len();
        let room = self.half.saturating_sub(self.head.len());
        let take = room.min(chunk.len());
        self.head.extend_from_slice(&chunk[..take]);
        chunk = &chunk[take..];
        if chunk.len() >= self.half {
            self.tail.clear();
            chunk = &chunk[chunk.len() - self.half..];
        }
        let overflow = (self.tail.len() + chunk.len()).saturating_sub(self.half);
        self.tail.drain(..overflow);
        self.tail.extend(chunk);
    }

    pub(crate) fn truncated(&self) -> bool {
        self.total > self.head.len() + self.tail.len()
    }

    pub(crate) fn render(&self) -> String {
        let tail: Vec<u8> = self.tail.iter().copied().collect();
        if !self.truncated() {
            return String::from_utf8_lossy(&[self.head.as_slice(), &tail].concat()).into_owned();
        }
        let omitted = self.total - self.head.len() - tail.len();
        format!(
            "{}\n[... {omitted} bytes omitted ...]\n{}",
            String::from_utf8_lossy(&self.head),
            String::from_utf8_lossy(&tail)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_short_output_whole() {
        let mut c = Capture::new(10);
        c.push(b"abc");
        c.push(b"de");
        assert!(!c.truncated());
        assert_eq!(c.render(), "abcde");
    }

    #[test]
    fn keeps_head_and_tail_of_long_output() {
        let mut c = Capture::new(8);
        for chunk in [b"0123".as_slice(), b"456789", b"ABCDEF"] {
            c.push(chunk);
        }
        assert!(c.truncated());
        assert_eq!(c.render(), "0123\n[... 8 bytes omitted ...]\nCDEF");
    }

    #[test]
    fn output_exactly_at_the_cap_is_not_truncated() {
        let mut c = Capture::new(8);
        c.push(b"01234567");
        assert!(!c.truncated());
        assert_eq!(c.render(), "01234567");
    }

    #[test]
    fn resolves_paths_against_the_workspace() {
        assert_eq!(resolve("src/main.rs").unwrap(), "/workspace/src/main.rs");
        assert_eq!(resolve("/tmp/x").unwrap(), "/tmp/x");
        assert!(resolve("  ").is_err());
        assert!(resolve("a\0b").is_err());
    }
}
