use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cairn_runtime::{Effect, Tool, ToolCtx, ToolError, ToolSpec};
use schemars::JsonSchema;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::sync::{Mutex, OnceCell};

use crate::sandbox::{Sandbox, SandboxError, SandboxProvider};

/// Limits on what the sandbox tools accept and return.
#[derive(Debug, Clone)]
pub struct SandboxOptions {
    /// Timeout for a command when the model does not ask for one.
    pub default_timeout: Duration,
    /// The longest timeout the model may ask for.
    pub max_timeout: Duration,
    /// Bytes of stdout and of stderr returned per command; the start and the
    /// end are kept.
    pub max_output: usize,
    /// Bytes returned by `read_file`.
    pub max_read: usize,
    /// Largest file `edit_file` will change.
    pub max_edit: usize,
    /// Overrides the effect of the `bash` tool. By default it is `Write`
    /// when the provider cuts commands off from the network and
    /// `Irreversible` (approval required by default) when it does not.
    pub bash_effect: Option<Effect>,
}

impl Default for SandboxOptions {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(60),
            max_timeout: Duration::from_secs(600),
            max_output: 32 * 1024,
            max_read: 256 * 1024,
            max_edit: 1024 * 1024,
            bash_effect: None,
        }
    }
}

type Slot = Arc<OnceCell<Arc<dyn Sandbox>>>;

/// Gives each run its own sandbox and exposes it to the model as tools.
///
/// ```ignore
/// let sandboxes = SandboxTools::new(DockerProvider::new("python:3.12-slim"), SandboxOptions::default());
/// let agent = Agent::builder("analyst").model(model).tools(sandboxes.tools()).build()?;
/// let outcome = agent.run_with_id(&run_id, "Load data.csv and plot revenue by month").await?;
/// sandboxes.close(&run_id).await?;
/// ```
///
/// A run's sandbox is created on its first tool call and kept until
/// [`SandboxTools::close`] (or the provider's lifetime cap). Runs never share
/// a sandbox.
pub struct SandboxTools {
    provider: Arc<dyn SandboxProvider>,
    options: SandboxOptions,
    open: Mutex<HashMap<String, Slot>>,
}

impl SandboxTools {
    pub fn new(provider: impl SandboxProvider + 'static, options: SandboxOptions) -> Arc<Self> {
        Arc::new(Self { provider: Arc::new(provider), options, open: Mutex::default() })
    }

    /// `bash`, `read_file`, `write_file` and `edit_file`.
    pub fn tools(self: &Arc<Self>) -> Vec<SandboxTool> {
        let bash_effect = self.options.bash_effect.unwrap_or(if self.provider.network_isolated() {
            Effect::Write
        } else {
            Effect::Irreversible
        });
        let max_timeout = self.options.max_timeout.as_secs();
        vec![
            self.tool::<BashInput>(
                Kind::Bash,
                "bash",
                &format!(
                    "Run a shell command (`sh -c`) in your private sandbox. The working directory is \
                     /workspace, which keeps its files between calls. Returns exit_code, stdout and stderr; \
                     long output keeps its start and end. The default timeout is {}s and the maximum is \
                     {max_timeout}s; on timeout every process the command started is killed, but files stay.{}",
                    self.options.default_timeout.as_secs(),
                    if self.provider.network_isolated() { " There is no network access." } else { "" },
                ),
                bash_effect,
                self.options.max_timeout + Duration::from_secs(60),
            ),
            self.tool::<ReadInput>(
                Kind::Read,
                "read_file",
                &format!(
                    "Read a text file from the sandbox. Relative paths are under /workspace. Returns at most \
                     {} bytes; `truncated` says when there is more (use `bash` with `sed -n` or `tail` for the rest).",
                    self.options.max_read
                ),
                Effect::Read,
                Duration::from_secs(180),
            ),
            self.tool::<WriteInput>(
                Kind::Write,
                "write_file",
                "Create or overwrite a file in the sandbox, creating parent directories. Relative paths are \
                 under /workspace.",
                Effect::Write,
                Duration::from_secs(180),
            ),
            self.tool::<EditInput>(
                Kind::Edit,
                "edit_file",
                "Replace text in a file in the sandbox. `old_text` must appear exactly once unless \
                 `replace_all` is true; include enough surrounding lines to make it unique.",
                Effect::Write,
                Duration::from_secs(180),
            ),
        ]
    }

    fn tool<I: JsonSchema>(
        self: &Arc<Self>,
        kind: Kind,
        name: &str,
        description: &str,
        effect: Effect,
        timeout: Duration,
    ) -> SandboxTool {
        let mut schema = serde_json::to_value(schemars::schema_for!(I)).unwrap_or(Value::Null);
        if let Value::Object(map) = &mut schema {
            map.remove("$schema");
            map.remove("title");
        }
        SandboxTool {
            tools: self.clone(),
            kind,
            timeout,
            spec: ToolSpec { name: name.into(), description: description.into(), input_schema: schema, effect },
        }
    }

    /// The sandbox for `run_id`, created if needed. Use it to put files in
    /// place before a run or to collect results after one.
    pub async fn sandbox(&self, run_id: &str) -> Result<Arc<dyn Sandbox>, SandboxError> {
        let slot = self.open.lock().await.entry(run_id.to_owned()).or_default().clone();
        slot.get_or_try_init(|| self.provider.open(run_id)).await.cloned()
    }

    /// Destroys the sandbox for `run_id` and its files.
    pub async fn close(&self, run_id: &str) -> Result<(), SandboxError> {
        let slot = self.open.lock().await.remove(run_id);
        match slot.as_ref().and_then(|s| s.get()) {
            Some(sandbox) => sandbox.close().await,
            // Not opened by this process (e.g. before a restart): reattach to close it.
            None => self.provider.open(run_id).await?.close().await,
        }
    }

    /// Destroys every sandbox this process has opened.
    pub async fn close_all(&self) -> Result<(), SandboxError> {
        let slots: Vec<Slot> = self.open.lock().await.drain().map(|(_, s)| s).collect();
        let mut first_error = None;
        for sandbox in slots.iter().filter_map(|s| s.get()) {
            if let Err(e) = sandbox.close().await {
                first_error.get_or_insert(e);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Forgets a sandbox that no longer exists, so the next call starts a fresh one.
    async fn forget(&self, run_id: &str, gone: &Arc<dyn Sandbox>) {
        let mut open = self.open.lock().await;
        if open.get(run_id).and_then(|s| s.get()).is_some_and(|s| Arc::ptr_eq(s, gone)) {
            open.remove(run_id);
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    Bash,
    Read,
    Write,
    Edit,
}

#[derive(Deserialize, JsonSchema)]
struct BashInput {
    /// The command, run with `sh -c` in /workspace.
    command: String,
    /// Seconds before the command is killed.
    timeout_secs: Option<u64>,
}

#[derive(Deserialize, JsonSchema)]
struct ReadInput {
    /// File path; relative paths are under /workspace.
    path: String,
}

#[derive(Deserialize, JsonSchema)]
struct WriteInput {
    /// File path; relative paths are under /workspace.
    path: String,
    /// The full new contents.
    content: String,
}

#[derive(Deserialize, JsonSchema)]
struct EditInput {
    /// File path; relative paths are under /workspace.
    path: String,
    /// The exact text to replace.
    old_text: String,
    /// The text to put in its place.
    new_text: String,
    /// Replace every occurrence instead of exactly one.
    replace_all: Option<bool>,
}

/// One of the sandbox tools; see [`SandboxTools::tools`].
pub struct SandboxTool {
    tools: Arc<SandboxTools>,
    kind: Kind,
    timeout: Duration,
    spec: ToolSpec,
}

fn parse<I: DeserializeOwned>(name: &str, input: Value) -> Result<I, ToolError> {
    serde_json::from_value(input).map_err(|e| ToolError(format!("invalid input for `{name}`: {e}")))
}

/// Text for the model, or an error when the bytes are not text.
fn decode(bytes: Vec<u8>, truncated: bool) -> Result<String, ToolError> {
    match String::from_utf8(bytes) {
        Ok(text) => Ok(text),
        // A cut in the middle of the last character is not binary data.
        Err(e) if truncated && e.utf8_error().valid_up_to() + 4 > e.as_bytes().len() => {
            let valid = e.utf8_error().valid_up_to();
            let mut bytes = e.into_bytes();
            bytes.truncate(valid);
            Ok(String::from_utf8(bytes).expect("checked prefix"))
        }
        Err(_) => Err(ToolError::new("this is a binary file; inspect it with `bash` (e.g. `file`, `xxd | head`)")),
    }
}

impl SandboxTool {
    async fn run(&self, sandbox: &Arc<dyn Sandbox>, input: Value) -> Result<Value, SandboxErrorOr> {
        let opts = &self.tools.options;
        let name = &self.spec.name;
        match self.kind {
            Kind::Bash => {
                let i: BashInput = parse(name, input)?;
                let timeout = i
                    .timeout_secs
                    .map(|s| Duration::from_secs(s.max(1)).min(opts.max_timeout))
                    .unwrap_or(opts.default_timeout);
                let out = sandbox.exec(&i.command, timeout, opts.max_output).await?;
                Ok(serde_json::to_value(out).expect("serializable"))
            }
            Kind::Read => {
                let i: ReadInput = parse(name, input)?;
                let file = sandbox.read_file(&i.path, opts.max_read).await?;
                let truncated = file.truncated();
                let content = decode(file.bytes, truncated)?;
                Ok(json!({"path": i.path, "size": file.size, "truncated": truncated, "content": content}))
            }
            Kind::Write => {
                let i: WriteInput = parse(name, input)?;
                sandbox.write_file(&i.path, i.content.as_bytes()).await?;
                Ok(json!({"path": i.path, "bytes": i.content.len()}))
            }
            Kind::Edit => {
                let i: EditInput = parse(name, input)?;
                if i.old_text.is_empty() {
                    return Err(ToolError::new("old_text is empty").into());
                }
                let file = sandbox.read_file(&i.path, opts.max_edit).await?;
                if file.truncated() {
                    return Err(ToolError(format!(
                        "{} is {} bytes, more than edit_file handles ({}); edit it with `bash`",
                        i.path, file.size, opts.max_edit
                    ))
                    .into());
                }
                let text = decode(file.bytes, false)?;
                let count = text.matches(&i.old_text).count();
                let replace_all = i.replace_all.unwrap_or(false);
                let updated = match count {
                    0 => return Err(ToolError(format!("old_text was not found in {}", i.path)).into()),
                    1 => text.replacen(&i.old_text, &i.new_text, 1),
                    _ if replace_all => text.replace(&i.old_text, &i.new_text),
                    n => {
                        return Err(ToolError(format!(
                            "old_text appears {n} times in {}; add surrounding lines to make it unique, or set replace_all",
                            i.path
                        ))
                        .into());
                    }
                };
                sandbox.write_file(&i.path, updated.as_bytes()).await?;
                Ok(json!({"path": i.path, "replacements": count}))
            }
        }
    }
}

/// A sandbox failure, kept apart from input errors so an expired sandbox
/// can be forgotten.
enum SandboxErrorOr {
    Sandbox(SandboxError),
    Tool(ToolError),
}

impl From<SandboxError> for SandboxErrorOr {
    fn from(e: SandboxError) -> Self {
        Self::Sandbox(e)
    }
}

impl From<ToolError> for SandboxErrorOr {
    fn from(e: ToolError) -> Self {
        Self::Tool(e)
    }
}

#[async_trait]
impl Tool for SandboxTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn timeout(&self) -> Option<Duration> {
        Some(self.timeout)
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
        let sandbox = self.tools.sandbox(&ctx.run_id).await.map_err(|e| ToolError(e.to_string()))?;
        match self.run(&sandbox, input).await {
            Ok(v) => Ok(v),
            Err(SandboxErrorOr::Tool(e)) => Err(e),
            Err(SandboxErrorOr::Sandbox(SandboxError::Gone(_))) => {
                self.tools.forget(&ctx.run_id, &sandbox).await;
                Err(ToolError::new(
                    "the sandbox expired and its files were lost; the next call starts a fresh, empty one",
                ))
            }
            Err(SandboxErrorOr::Sandbox(e)) => Err(ToolError(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;
    use crate::sandbox::{ExecOutput, FileContents, resolve};

    #[derive(Default)]
    struct MemSandbox {
        files: std::sync::Mutex<HashMap<String, Vec<u8>>>,
        gone: AtomicBool,
        closed: AtomicBool,
    }

    #[async_trait]
    impl Sandbox for MemSandbox {
        async fn exec(&self, command: &str, timeout: Duration, _max: usize) -> Result<ExecOutput, SandboxError> {
            if self.gone.load(Ordering::SeqCst) {
                return Err(SandboxError::Gone("mem".into()));
            }
            Ok(ExecOutput {
                exit_code: Some(0),
                stdout: format!("{command} ({}s)", timeout.as_secs()),
                stderr: String::new(),
                truncated: false,
                timed_out: false,
                duration_ms: 0,
            })
        }

        async fn write_file(&self, path: &str, contents: &[u8]) -> Result<(), SandboxError> {
            self.files.lock().unwrap().insert(resolve(path)?, contents.to_vec());
            Ok(())
        }

        async fn read_file(&self, path: &str, max: usize) -> Result<FileContents, SandboxError> {
            let path = resolve(path)?;
            let files = self.files.lock().unwrap();
            let bytes = files.get(&path).ok_or_else(|| SandboxError::File(format!("no such file: {path}")))?;
            Ok(FileContents { bytes: bytes[..bytes.len().min(max)].to_vec(), size: bytes.len() as u64 })
        }

        async fn close(&self) -> Result<(), SandboxError> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Default)]
    struct MemProvider {
        opened: AtomicUsize,
        last: std::sync::Mutex<Option<Arc<MemSandbox>>>,
        networked: bool,
    }

    #[async_trait]
    impl SandboxProvider for Arc<MemProvider> {
        async fn open(&self, _key: &str) -> Result<Arc<dyn Sandbox>, SandboxError> {
            self.opened.fetch_add(1, Ordering::SeqCst);
            let sandbox = Arc::new(MemSandbox::default());
            *self.last.lock().unwrap() = Some(sandbox.clone());
            Ok(sandbox)
        }

        fn network_isolated(&self) -> bool {
            !self.networked
        }
    }

    fn setup(networked: bool) -> (Arc<MemProvider>, Arc<SandboxTools>, HashMap<String, SandboxTool>) {
        let provider = Arc::new(MemProvider { networked, ..Default::default() });
        let tools = SandboxTools::new(provider.clone(), SandboxOptions::default());
        let by_name = tools.tools().into_iter().map(|t| (t.spec.name.clone(), t)).collect();
        (provider, tools, by_name)
    }

    #[tokio::test]
    async fn files_round_trip_and_edits_need_a_unique_match() {
        let (_, _, t) = setup(false);
        let ctx = ToolCtx::new("run-1", "c1");
        t["write_file"].call(json!({"path": "a.txt", "content": "x = 1\ny = 1\n"}), &ctx).await.unwrap();
        let read = t["read_file"].call(json!({"path": "/workspace/a.txt"}), &ctx).await.unwrap();
        assert_eq!(read["content"], "x = 1\ny = 1\n");
        assert_eq!(read["truncated"], false);

        let ambiguous = t["edit_file"].call(json!({"path": "a.txt", "old_text": "= 1", "new_text": "= 2"}), &ctx).await;
        assert!(ambiguous.unwrap_err().0.contains("appears 2 times"));
        let missing = t["edit_file"].call(json!({"path": "a.txt", "old_text": "z", "new_text": "w"}), &ctx).await;
        assert!(missing.unwrap_err().0.contains("not found"));

        t["edit_file"].call(json!({"path": "a.txt", "old_text": "x = 1", "new_text": "x = 3"}), &ctx).await.unwrap();
        let all = json!({"path": "a.txt", "old_text": "= ", "new_text": ": ", "replace_all": true});
        assert_eq!(t["edit_file"].call(all, &ctx).await.unwrap()["replacements"], 2);
        let read = t["read_file"].call(json!({"path": "a.txt"}), &ctx).await.unwrap();
        assert_eq!(read["content"], "x : 3\ny : 1\n");
    }

    #[tokio::test]
    async fn each_run_gets_its_own_sandbox_once() {
        let (provider, tools, t) = setup(false);
        let (a, b) = (ToolCtx::new("run-a", "1"), ToolCtx::new("run-b", "1"));
        t["write_file"].call(json!({"path": "f", "content": "a"}), &a).await.unwrap();
        t["bash"].call(json!({"command": "true"}), &a).await.unwrap();
        assert!(t["read_file"].call(json!({"path": "f"}), &b).await.is_err(), "runs must not share files");
        assert_eq!(provider.opened.load(Ordering::SeqCst), 2);

        let sandbox_b = provider.last.lock().unwrap().clone().unwrap();
        tools.close("run-b").await.unwrap();
        assert!(sandbox_b.closed.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn clamps_timeouts() {
        let (_, _, t) = setup(false);
        let ctx = ToolCtx::new("r", "1");
        let out = t["bash"].call(json!({"command": "x", "timeout_secs": 99999}), &ctx).await.unwrap();
        assert_eq!(out["stdout"], "x (600s)");
        let out = t["bash"].call(json!({"command": "x"}), &ctx).await.unwrap();
        assert_eq!(out["stdout"], "x (60s)");
    }

    #[tokio::test]
    async fn an_expired_sandbox_is_replaced_on_the_next_call() {
        let (provider, _, t) = setup(false);
        let ctx = ToolCtx::new("r", "1");
        t["bash"].call(json!({"command": "x"}), &ctx).await.unwrap();
        provider.last.lock().unwrap().as_ref().unwrap().gone.store(true, Ordering::SeqCst);
        let err = t["bash"].call(json!({"command": "x"}), &ctx).await.unwrap_err();
        assert!(err.0.contains("expired"));
        t["bash"].call(json!({"command": "x"}), &ctx).await.unwrap();
        assert_eq!(provider.opened.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn bash_needs_approval_when_the_network_is_open() {
        assert_eq!(setup(false).2["bash"].spec.effect, Effect::Write);
        assert_eq!(setup(true).2["bash"].spec.effect, Effect::Irreversible);
        assert_eq!(setup(true).2["read_file"].spec.effect, Effect::Read);
    }

    #[test]
    fn decodes_text_cut_mid_character() {
        let mut bytes = "héllo".as_bytes().to_vec();
        bytes.truncate(2); // cuts `é` in half
        assert_eq!(decode(bytes.clone(), true).unwrap(), "h");
        assert!(decode(bytes, false).is_err());
        assert!(decode(vec![0xff, 0xfe, 0, 1, 2, 3, 4, 5], true).is_err());
    }
}
