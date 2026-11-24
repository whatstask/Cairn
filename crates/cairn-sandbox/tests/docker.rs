//! Runs against a real container engine. Skipped when Docker is not
//! available, unless `CAIRN_SANDBOX_TESTS=1` (set in CI), which makes a
//! missing engine a failure.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cairn_runtime::{Tool, ToolCtx};
use cairn_sandbox::{DockerProvider, Sandbox, SandboxError, SandboxOptions, SandboxProvider, SandboxTools};
use serde_json::json;

const IMAGE: &str = "alpine:3.20";
const T: Duration = Duration::from_secs(30);

async fn docker() -> bool {
    let up = tokio::process::Command::new("docker")
        .args(["version", "--format", "{{.Server.Version}}"])
        .output()
        .await
        .is_ok_and(|o| o.status.success());
    if !up {
        assert!(
            std::env::var("CAIRN_SANDBOX_TESTS").as_deref() != Ok("1"),
            "CAIRN_SANDBOX_TESTS=1 but Docker is not available"
        );
        eprintln!("skipping: Docker is not available");
    }
    up
}

fn key(name: &str) -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    format!("test-{name}-{}-{nanos}", std::process::id())
}

fn provider() -> DockerProvider {
    DockerProvider::new(IMAGE).max_lifetime(Duration::from_secs(600))
}

async fn sh(sandbox: &Arc<dyn Sandbox>, command: &str) -> (Option<i32>, String, String) {
    let out = sandbox.exec(command, T, 64 * 1024).await.unwrap();
    (out.exit_code, out.stdout.trim().to_owned(), out.stderr.trim().to_owned())
}

#[tokio::test]
async fn commands_run_unprivileged_offline_and_read_only() {
    if !docker().await {
        return;
    }
    let sandbox = provider().open(&key("isolation")).await.unwrap();

    assert_eq!(sh(&sandbox, "id -u").await.1, "65534");
    assert_eq!(sh(&sandbox, "pwd").await.1, "/workspace");
    let (code, out, err) = sh(&sandbox, "echo out; echo err >&2; exit 3").await;
    assert_eq!((code, out.as_str(), err.as_str()), (Some(3), "out", "err"));

    // Only loopback: no route to the host or the internet.
    assert_eq!(sh(&sandbox, "ls /sys/class/net").await.1, "lo");
    assert_ne!(sh(&sandbox, "wget -q -T 3 -O- http://1.1.1.1").await.0, Some(0));
    // The image is read-only; only the workspace and /tmp are writable.
    assert_ne!(sh(&sandbox, "touch /etc/cairn").await.0, Some(0));
    assert_eq!(sh(&sandbox, "touch /tmp/x && echo ok").await.1, "ok");
    // No capabilities, and no way to gain them.
    assert_eq!(sh(&sandbox, "grep CapEff /proc/self/status").await.1, "CapEff:\t0000000000000000");
    assert_eq!(sh(&sandbox, "grep NoNewPrivs /proc/self/status").await.1, "NoNewPrivs:\t1");

    sandbox.write_file("dir/a.txt", b"hello\n").await.unwrap();
    assert_eq!(sh(&sandbox, "cat dir/a.txt").await.1, "hello");
    let file = sandbox.read_file("/workspace/dir/a.txt", 3).await.unwrap();
    assert_eq!((file.bytes.as_slice(), file.size), (b"hel".as_slice(), 6));
    assert!(matches!(sandbox.read_file("nope", 10).await, Err(SandboxError::File(_))));

    sandbox.close().await.unwrap();
}

#[tokio::test]
async fn a_timeout_kills_everything_the_command_started_but_keeps_files() {
    if !docker().await {
        return;
    }
    let sandbox = provider().open(&key("timeout")).await.unwrap();
    sandbox.write_file("keep.txt", b"kept").await.unwrap();

    let out = sandbox.exec("sleep 300 & echo $! > pid; wait", Duration::from_secs(2), 1024).await.unwrap();
    assert!(out.timed_out);
    assert_eq!(out.exit_code, None);

    assert_eq!(sh(&sandbox, "kill -0 $(cat pid) 2>/dev/null && echo alive || echo dead").await.1, "dead");
    assert_eq!(sh(&sandbox, "cat keep.txt").await.1, "kept");
    sandbox.close().await.unwrap();
}

#[tokio::test]
async fn long_output_keeps_its_start_and_end() {
    if !docker().await {
        return;
    }
    let sandbox = provider().open(&key("output")).await.unwrap();
    let out = sandbox.exec("echo START; head -c 100000 /dev/zero | tr '\\0' x; echo; echo END", T, 1000).await.unwrap();
    assert!(out.truncated);
    assert!(out.stdout.starts_with("START") && out.stdout.trim_end().ends_with("END"));
    assert!(out.stdout.len() < 1100);
    sandbox.close().await.unwrap();
}

#[tokio::test]
async fn process_and_disk_limits_hold() {
    if !docker().await {
        return;
    }
    let sandbox = provider().pids(32).workspace_mb(8).open(&key("limits")).await.unwrap();

    let forks = sandbox.exec("for i in $(seq 1 64); do sleep 2 & done; wait", T, 4096).await.unwrap();
    assert!(!forks.stderr.is_empty(), "expected fork failures past the process limit");
    let (code, _, err) = sh(&sandbox, "head -c 20000000 /dev/zero > big").await;
    assert_ne!(code, Some(0));
    assert!(err.contains("No space left"), "{err}");
    sandbox.close().await.unwrap();
}

#[tokio::test]
async fn a_resumed_run_reattaches_and_a_closed_sandbox_is_gone() {
    if !docker().await {
        return;
    }
    let k = key("reattach");
    let first = provider().open(&k).await.unwrap();
    first.write_file("state", b"1").await.unwrap();

    // A different provider, as after a process restart, finds the same workspace.
    let second = provider().open(&k).await.unwrap();
    assert_eq!(sh(&second, "cat state").await.1, "1");

    second.close().await.unwrap();
    assert!(matches!(first.exec("true", T, 100).await, Err(SandboxError::Gone(_))));
    first.close().await.unwrap();
}

#[tokio::test]
async fn agents_use_it_through_tools() {
    if !docker().await {
        return;
    }
    let tools = SandboxTools::new(provider(), SandboxOptions::default());
    let by_name: std::collections::HashMap<String, _> =
        tools.tools().into_iter().map(|t| (t.spec().name.clone(), t)).collect();
    let run = key("tools");
    let ctx = ToolCtx::new(&run, "call-1");

    by_name["write_file"].call(json!({"path": "app.py", "content": "print(1 + 1)\n"}), &ctx).await.unwrap();
    by_name["edit_file"]
        .call(json!({"path": "app.py", "old_text": "1 + 1", "new_text": "2 * 21"}), &ctx)
        .await
        .unwrap();
    let out = by_name["bash"].call(json!({"command": "awk '{print}' app.py && echo done"}), &ctx).await.unwrap();
    assert_eq!(out["exit_code"], 0);
    assert_eq!(out["stdout"], "print(2 * 21)\ndone\n");

    tools.close(&run).await.unwrap();
    let err = by_name["bash"].call(json!({"command": "ls app.py"}), &ctx).await.unwrap();
    assert_ne!(err["exit_code"], 0, "a closed run starts over with an empty workspace");
    tools.close(&run).await.unwrap();
}
