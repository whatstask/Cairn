use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use crate::event::StrategyEvent;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sequence conflict for program {program}: expected {expected}, got {got}")]
    Conflict { program: String, expected: u64, got: u64 },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt store entry: {0}")]
    Corrupt(String),
    #[error("store backend error: {0}")]
    Backend(String),
}

/// Append-only, per-program event storage. Implementations must reject an
/// append whose `seq` is not exactly the next position, which keeps
/// concurrent engines from forking a program's history.
#[async_trait]
pub trait StrategyStore: Send + Sync {
    async fn append(&self, event: &StrategyEvent) -> Result<(), StoreError>;
    async fn load(&self, program: &str) -> Result<Vec<StrategyEvent>, StoreError>;
    async fn list_programs(&self) -> Result<Vec<String>, StoreError>;
}

#[derive(Default)]
pub struct InMemoryStore {
    programs: Mutex<HashMap<String, Vec<StrategyEvent>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl StrategyStore for InMemoryStore {
    async fn append(&self, event: &StrategyEvent) -> Result<(), StoreError> {
        let mut programs = self.programs.lock().await;
        let events = programs.entry(event.program.clone()).or_default();
        let expected = events.len() as u64;
        if event.seq != expected {
            return Err(StoreError::Conflict { program: event.program.clone(), expected, got: event.seq });
        }
        events.push(event.clone());
        Ok(())
    }

    async fn load(&self, program: &str) -> Result<Vec<StrategyEvent>, StoreError> {
        Ok(self.programs.lock().await.get(program).cloned().unwrap_or_default())
    }

    async fn list_programs(&self) -> Result<Vec<String>, StoreError> {
        let mut names: Vec<String> = self.programs.lock().await.keys().cloned().collect();
        names.sort();
        Ok(names)
    }
}

/// One JSON-lines file per program under a directory.
pub struct JsonlStore {
    dir: PathBuf,
    /// Next expected seq per program, loaded lazily.
    next: Mutex<HashMap<String, u64>>,
}

impl JsonlStore {
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, next: Mutex::new(HashMap::new()) })
    }

    fn path(&self, program: &str) -> PathBuf {
        let safe: String =
            program.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        self.dir.join(format!("{safe}.jsonl"))
    }
}

#[async_trait]
impl StrategyStore for JsonlStore {
    async fn append(&self, event: &StrategyEvent) -> Result<(), StoreError> {
        let mut next = self.next.lock().await;
        let expected = match next.get(&event.program) {
            Some(n) => *n,
            None => self.load(&event.program).await?.len() as u64,
        };
        if event.seq != expected {
            return Err(StoreError::Conflict { program: event.program.clone(), expected, got: event.seq });
        }
        let mut line = serde_json::to_vec(event).map_err(|e| StoreError::Corrupt(e.to_string()))?;
        line.push(b'\n');
        let mut f = tokio::fs::OpenOptions::new().create(true).append(true).open(self.path(&event.program)).await?;
        f.write_all(&line).await?;
        f.flush().await?;
        next.insert(event.program.clone(), expected + 1);
        Ok(())
    }

    async fn load(&self, program: &str) -> Result<Vec<StrategyEvent>, StoreError> {
        let text = match tokio::fs::read_to_string(self.path(program)).await {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(|e| StoreError::Corrupt(e.to_string())))
            .collect()
    }

    async fn list_programs(&self) -> Result<Vec<String>, StoreError> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "jsonl") {
                if let Some(first) = self.load_first(&path)? {
                    names.push(first.program);
                }
            }
        }
        names.sort();
        Ok(names)
    }
}

impl JsonlStore {
    fn load_first(&self, path: &std::path::Path) -> Result<Option<StrategyEvent>, StoreError> {
        let text = std::fs::read_to_string(path)?;
        match text.lines().find(|l| !l.trim().is_empty()) {
            Some(l) => serde_json::from_str(l).map(Some).map_err(|e| StoreError::Corrupt(e.to_string())),
            None => Ok(None),
        }
    }
}
