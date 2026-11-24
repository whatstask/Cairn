use std::collections::BTreeMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::sandbox::{Capture, ExecOutput, FileContents, Sandbox, SandboxError, SandboxProvider, WORKSPACE, resolve};

/// What a sandbox's commands can reach over the network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Network {
    /// No network at all, not even the host (the default).
    None,
    /// Docker's default bridge: full outbound access. Only for trusted work.
    Bridge,
    /// A Docker network you manage, e.g. an internal network whose only
    /// route out is an egress proxy with an allowlist.
    Named(String),
}

/// Sandboxes as hardened containers, driven through the Docker CLI (or any
/// CLI with the same interface, such as Podman).
///
/// Every container runs with no network (by default), a read-only root
/// filesystem, all Linux capabilities dropped, `no-new-privileges`, and
/// limits on memory, CPU and process count. Writable space is two size-capped
/// in-memory mounts: [`WORKSPACE`] and `/tmp`. Commands run as an
/// unprivileged user (65534 by default). The container removes itself when
/// its lifetime cap is reached, even if this process has crashed.
///
/// For untrusted code in production, also run the containers under gVisor
/// with [`DockerProvider::runtime`]`("runsc")`, which puts a user-space kernel
/// between the code and the host.
///
/// The image needs a POSIX shell and the usual basic utilities (`sleep`,
/// `kill`, `cat`, `head`, `wc`, `mkdir`, `dirname`); any Debian, Ubuntu,
/// Alpine or language image has them.
#[derive(Debug, Clone)]
pub struct DockerProvider {
    cli: String,
    image: String,
    runtime: Option<String>,
    network: Network,
    memory_mb: u64,
    cpus: f64,
    pids: u32,
    workspace_mb: u64,
    tmp_mb: u64,
    uid: u32,
    gid: u32,
    env: BTreeMap<String, String>,
    max_lifetime: Duration,
    start_timeout: Duration,
    prefix: String,
}

impl DockerProvider {
    /// Sandboxes from `image`, e.g. `"python:3.12-slim"`. Pin a digest in
    /// production (`python@sha256:…`) so the environment cannot change under you.
    pub fn new(image: &str) -> Self {
        Self {
            cli: "docker".into(),
            image: image.into(),
            runtime: None,
            network: Network::None,
            memory_mb: 1024,
            cpus: 1.0,
            pids: 256,
            workspace_mb: 1024,
            tmp_mb: 256,
            uid: 65534,
            gid: 65534,
            env: BTreeMap::new(),
            max_lifetime: Duration::from_secs(3600),
            start_timeout: Duration::from_secs(300),
            prefix: "cairn-sbx".into(),
        }
    }

    /// The container CLI (default `docker`; `podman` works too).
    pub fn cli(mut self, cli: &str) -> Self {
        self.cli = cli.into();
        self
    }

    /// OCI runtime, e.g. `"runsc"` for gVisor.
    pub fn runtime(mut self, runtime: &str) -> Self {
        self.runtime = Some(runtime.into());
        self
    }

    pub fn network(mut self, network: Network) -> Self {
        self.network = network;
        self
    }

    /// Memory limit in MiB; swap is disabled.
    pub fn memory_mb(mut self, mb: u64) -> Self {
        self.memory_mb = mb;
        self
    }

    pub fn cpus(mut self, cpus: f64) -> Self {
        self.cpus = cpus;
        self
    }

    /// Maximum number of processes and threads.
    pub fn pids(mut self, pids: u32) -> Self {
        self.pids = pids;
        self
    }

    /// Size of the workspace in MiB. It is held in memory and counts toward
    /// the host's memory, not the container's limit.
    pub fn workspace_mb(mut self, mb: u64) -> Self {
        self.workspace_mb = mb;
        self
    }

    /// Size of `/tmp` in MiB.
    pub fn tmp_mb(mut self, mb: u64) -> Self {
        self.tmp_mb = mb;
        self
    }

    /// The user commands run as. Never 0: commands must not be able to
    /// signal the container's keep-alive process.
    pub fn user(mut self, uid: u32, gid: u32) -> Self {
        assert!(uid != 0, "sandbox commands must not run as root");
        self.uid = uid;
        self.gid = gid;
        self
    }

    /// An environment variable for every command. Not for secrets: anything
    /// here is readable by the code the agent runs.
    pub fn env(mut self, key: &str, value: &str) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// The container removes itself after this long, whatever happens to
    /// this process. Runs that outlive it get a fresh, empty workspace.
    pub fn max_lifetime(mut self, lifetime: Duration) -> Self {
        self.max_lifetime = lifetime;
        self
    }

    /// How long creating a container may take, including pulling the image.
    pub fn start_timeout(mut self, timeout: Duration) -> Self {
        self.start_timeout = timeout;
        self
    }

    /// Container name prefix (default `cairn-sbx`).
    pub fn name_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.into();
        self
    }

    fn name_for(&self, key: &str) -> String {
        let digest = Sha256::digest(key.as_bytes());
        format!("{}-{}", self.prefix, hex::encode(&digest[..16]))
    }

    fn run_args(&self, name: &str) -> Vec<String> {
        let network = match &self.network {
            Network::None => "none".to_owned(),
            Network::Bridge => "bridge".to_owned(),
            Network::Named(n) => n.clone(),
        };
        let (uid, gid) = (self.uid, self.gid);
        let mut args: Vec<String> = [
            "run",
            "--detach",
            "--rm",
            "--init",
            "--name",
            name,
            // The keep-alive process runs as root with no capabilities, so the
            // unprivileged commands cannot kill it when they are cleaned up.
            "--user",
            "0:0",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--label",
            "cairn.sandbox=1",
            "--pull",
            "missing",
            "--entrypoint",
            "sleep",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        args.extend([
            "--network".into(),
            network,
            "--pids-limit".into(),
            self.pids.to_string(),
            "--memory".into(),
            format!("{}m", self.memory_mb),
            "--memory-swap".into(),
            format!("{}m", self.memory_mb),
            "--cpus".into(),
            self.cpus.to_string(),
            "--tmpfs".into(),
            format!("{WORKSPACE}:rw,exec,nosuid,nodev,size={}m,uid={uid},gid={gid},mode=0700", self.workspace_mb),
            "--tmpfs".into(),
            format!("/tmp:rw,exec,nosuid,nodev,size={}m,mode=1777", self.tmp_mb),
            "--env".into(),
            format!("HOME={WORKSPACE}"),
        ]);
        for (k, v) in &self.env {
            args.extend(["--env".into(), format!("{k}={v}")]);
        }
        if let Some(runtime) = &self.runtime {
            args.extend(["--runtime".into(), runtime.clone()]);
        }
        args.extend([self.image.clone(), self.max_lifetime.as_secs().max(1).to_string()]);
        args
    }

    async fn state(&self, name: &str) -> Result<Option<bool>, SandboxError> {
        let out = cli(&self.cli, &["inspect", "--format", "{{.State.Running}}", name], Duration::from_secs(30)).await?;
        if out.status.success() {
            return Ok(Some(String::from_utf8_lossy(&out.stdout).trim() == "true"));
        }
        let err = String::from_utf8_lossy(&out.stderr);
        if is_missing(&err) { Ok(None) } else { Err(SandboxError::Backend(err.trim().to_owned())) }
    }
}

#[async_trait]
impl SandboxProvider for DockerProvider {
    async fn open(&self, key: &str) -> Result<Arc<dyn Sandbox>, SandboxError> {
        let name = self.name_for(key);
        let sandbox = DockerSandbox {
            cli: self.cli.clone(),
            name: name.clone(),
            user: format!("{}:{}", self.uid, self.gid),
            turn: Arc::new(Mutex::new(())),
        };
        for _ in 0..3 {
            match self.state(&name).await? {
                Some(true) => return Ok(Arc::new(sandbox)),
                Some(false) => {
                    // Stopped but not yet removed: clear it and start fresh.
                    cli(&self.cli, &["rm", "--force", &name], Duration::from_secs(60)).await?;
                }
                None => {}
            }
            let args = self.run_args(&name);
            let args: Vec<&str> = args.iter().map(String::as_str).collect();
            let out = cli(&self.cli, &args, self.start_timeout).await?;
            if out.status.success() {
                tracing::info!(sandbox = %name, image = %self.image, "sandbox started");
                return Ok(Arc::new(sandbox));
            }
            let err = String::from_utf8_lossy(&out.stderr);
            // Another process created it first; loop to reattach.
            if !(err.contains("Conflict") || err.contains("already in use")) {
                return Err(SandboxError::Backend(format!("could not start sandbox: {}", err.trim())));
            }
        }
        Err(SandboxError::Backend(format!("could not start or reattach to sandbox `{name}`")))
    }

    fn network_isolated(&self) -> bool {
        self.network == Network::None
    }
}

/// One container.
#[derive(Debug)]
pub struct DockerSandbox {
    cli: String,
    name: String,
    user: String,
    /// One operation at a time: cleaning up after a timeout kills every
    /// command in the sandbox, so commands must not overlap.
    turn: Arc<Mutex<()>>,
}

impl DockerSandbox {
    /// The container's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn command(&self, interactive: bool, args: &[&str]) -> Command {
        let mut cmd = Command::new(&self.cli);
        cmd.arg("exec");
        if interactive {
            cmd.arg("--interactive");
        }
        cmd.args(["--user", &self.user, "--workdir", WORKSPACE, &self.name]).args(args);
        cmd
    }

    fn failure(&self, stderr: &[u8]) -> SandboxError {
        let err = String::from_utf8_lossy(stderr);
        if is_missing(&err) { SandboxError::Gone(self.name.clone()) } else { SandboxError::File(err.trim().to_owned()) }
    }
}

#[async_trait]
impl Sandbox for DockerSandbox {
    async fn exec(&self, command: &str, timeout: Duration, max_output: usize) -> Result<ExecOutput, SandboxError> {
        let turn = self.turn.clone().lock_owned().await;
        let started = Instant::now();
        let mut child = self
            .command(false, &["sh", "-c", command])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| SandboxError::Backend(format!("could not run `{}`: {e}", self.cli)))?;
        // If this future is dropped mid-command (the run was cancelled), the
        // guard still kills what the command started, and holds the sandbox
        // until it has, so the cleanup cannot hit the next command.
        let mut guard = KillGuard {
            cli: self.cli.clone(),
            name: self.name.clone(),
            user: self.user.clone(),
            turn: Some(turn),
            armed: true,
        };
        let (mut stdout, mut stderr) = (Capture::new(max_output), Capture::new(max_output));
        let (out_pipe, err_pipe) = (child.stdout.take(), child.stderr.take());
        let finished = tokio::time::timeout(timeout, async {
            tokio::join!(drain(out_pipe, &mut stdout), drain(err_pipe, &mut stderr));
            child.wait().await
        })
        .await;
        let (exit_code, timed_out) = match finished {
            Ok(Ok(status)) => (status.code(), false),
            Ok(Err(e)) => return Err(SandboxError::Backend(format!("waiting for command: {e}"))),
            Err(_) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                kill_all(&self.cli, &self.name, &self.user).await;
                (None, true)
            }
        };
        guard.armed = false;
        let stderr_text = stderr.render();
        if !timed_out && exit_code != Some(0) && is_missing(&stderr_text) {
            return Err(SandboxError::Gone(self.name.clone()));
        }
        Ok(ExecOutput {
            exit_code,
            truncated: stdout.truncated() || stderr.truncated(),
            stdout: stdout.render(),
            stderr: stderr_text,
            timed_out,
            duration_ms: started.elapsed().as_millis() as u64,
        })
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> Result<(), SandboxError> {
        let path = resolve(path)?;
        let _turn = self.turn.lock().await;
        let script = r#"mkdir -p "$(dirname "$1")" && cat > "$1""#;
        let mut child = self
            .command(true, &["sh", "-c", script, "sh", &path])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| SandboxError::Backend(format!("could not run `{}`: {e}", self.cli)))?;
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let data = contents.to_vec();
        let writer = tokio::spawn(async move {
            let result = stdin.write_all(&data).await;
            drop(stdin);
            result
        });
        let out = tokio::time::timeout(Duration::from_secs(120), child.wait_with_output())
            .await
            .map_err(|_| SandboxError::Backend("writing the file timed out".into()))?
            .map_err(|e| SandboxError::Backend(e.to_string()))?;
        let written = writer.await.map_err(|e| SandboxError::Backend(e.to_string()))?;
        if !out.status.success() {
            return Err(self.failure(&out.stderr));
        }
        written.map_err(|e| SandboxError::Backend(format!("sending the file: {e}")))
    }

    async fn read_file(&self, path: &str, max_bytes: usize) -> Result<FileContents, SandboxError> {
        let path = resolve(path)?;
        let _turn = self.turn.lock().await;
        let script = r#"test -f "$1" || { echo "no such file: $1" >&2; exit 2; }; wc -c < "$1" && head -c "$2" "$1""#;
        let max = max_bytes.to_string();
        let out = tokio::time::timeout(
            Duration::from_secs(120),
            self.command(false, &["sh", "-c", script, "sh", &path, &max]).kill_on_drop(true).output(),
        )
        .await
        .map_err(|_| SandboxError::Backend("reading the file timed out".into()))?
        .map_err(|e| SandboxError::Backend(format!("could not run `{}`: {e}", self.cli)))?;
        if !out.status.success() {
            return Err(self.failure(&out.stderr));
        }
        let newline = out
            .stdout
            .iter()
            .position(|b| *b == b'\n')
            .ok_or_else(|| SandboxError::Backend("unexpected output while reading the file".into()))?;
        let size = String::from_utf8_lossy(&out.stdout[..newline])
            .trim()
            .parse()
            .map_err(|_| SandboxError::Backend("unexpected output while reading the file".into()))?;
        Ok(FileContents { bytes: out.stdout[newline + 1..].to_vec(), size })
    }

    async fn close(&self) -> Result<(), SandboxError> {
        let out = cli(&self.cli, &["rm", "--force", &self.name], Duration::from_secs(60)).await?;
        let err = String::from_utf8_lossy(&out.stderr);
        if out.status.success() || is_missing(&err) {
            Ok(())
        } else {
            Err(SandboxError::Backend(err.trim().to_owned()))
        }
    }
}

/// Kills the sandbox's commands if an `exec` is dropped before it finishes.
struct KillGuard {
    cli: String,
    name: String,
    user: String,
    turn: Option<OwnedMutexGuard<()>>,
    armed: bool,
}

impl Drop for KillGuard {
    fn drop(&mut self) {
        let turn = self.turn.take();
        if !self.armed {
            return;
        }
        let (cli, name, user) = (self.cli.clone(), self.name.clone(), self.user.clone());
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                kill_all(&cli, &name, &user).await;
                drop(turn);
            });
        }
    }
}

/// Kills every process the sandbox user owns. The keep-alive process belongs
/// to root, and the user has no capabilities, so the container survives.
async fn kill_all(cli_name: &str, name: &str, user: &str) {
    let result = Command::new(cli_name)
        .args(["exec", "--user", user, name, "sh", "-c", "kill -9 -1"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    if let Err(e) = result {
        tracing::warn!(sandbox = %name, error = %e, "could not clean up after a timed-out command");
    }
}

async fn drain(pipe: Option<impl AsyncRead + Unpin>, capture: &mut Capture) {
    let Some(mut pipe) = pipe else { return };
    let mut buf = [0u8; 8192];
    while let Ok(n) = pipe.read(&mut buf).await {
        if n == 0 {
            break;
        }
        capture.push(&buf[..n]);
    }
}

/// Runs the container CLI with a deadline.
async fn cli(cli_name: &str, args: &[&str], timeout: Duration) -> Result<std::process::Output, SandboxError> {
    let run = Command::new(cli_name).args(args).stdin(Stdio::null()).kill_on_drop(true).output();
    tokio::time::timeout(timeout, run)
        .await
        .map_err(|_| SandboxError::Backend(format!("`{cli_name} {}` timed out", args.first().unwrap_or(&""))))?
        .map_err(|e| SandboxError::Backend(format!("could not run `{cli_name}`: {e}")))
}

/// Whether CLI output says the container does not exist or has stopped.
fn is_missing(stderr: &str) -> bool {
    // CLIs word this differently across versions ("Error response from
    // daemon: No such container", "error: no such object", Podman's "Error:
    // no container with name or ID"), so match loosely but only on CLI errors.
    let s = stderr.trim_start().to_ascii_lowercase();
    s.starts_with("error")
        && ["no such container", "no such object", "is not running", "no container with name or id"]
            .iter()
            .any(|m| s.contains(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_stable_and_distinct_per_key() {
        let p = DockerProvider::new("alpine:3.20");
        assert_eq!(p.name_for("run-1"), p.name_for("run-1"));
        assert_ne!(p.name_for("run-1"), p.name_for("run-2"));
        assert!(p.name_for("run-1").starts_with("cairn-sbx-"));
    }

    #[test]
    fn hardens_the_container() {
        let args = DockerProvider::new("alpine:3.20").run_args("n").join(" ");
        for flag in [
            "--network none",
            "--read-only",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "--pids-limit 256",
            "--memory 1024m",
            "--memory-swap 1024m",
            "--user 0:0",
            "--rm",
            "--init",
        ] {
            assert!(args.contains(flag), "missing `{flag}` in {args}");
        }
        assert!(args.contains("/workspace:rw,exec,nosuid,nodev,size=1024m,uid=65534,gid=65534,mode=0700"));
        assert!(args.ends_with("alpine:3.20 3600"));
    }

    #[test]
    fn network_and_runtime_are_configurable() {
        let p = DockerProvider::new("img").network(Network::Named("egress".into())).runtime("runsc");
        let args = p.run_args("n").join(" ");
        assert!(args.contains("--network egress") && args.contains("--runtime runsc"));
        assert!(!p.network_isolated());
        assert!(DockerProvider::new("img").network_isolated());
    }

    #[test]
    #[should_panic(expected = "must not run as root")]
    fn refuses_root_commands() {
        let _ = DockerProvider::new("img").user(0, 0);
    }

    #[test]
    fn recognises_missing_containers() {
        assert!(is_missing("Error response from daemon: No such container: x"));
        assert!(is_missing("Error response from daemon: container abc is not running"));
        assert!(is_missing("Error: no container with name or ID \"x\" found"));
        assert!(is_missing("error: no such object: cairn-sbx-1"), "Docker CLI 28+ wording");
        assert!(!is_missing("No such container"), "a command's own output is not a CLI error");
        assert!(!is_missing("Error response from daemon: pull access denied"));
    }
}
