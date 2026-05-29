use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub db: DbConfig,
    pub embed: EmbedConfig,
    pub recall: RecallConfig,
    pub content: ContentConfig,
    pub classification: ClassificationConfig,
    pub ppr: PprConfig,
    pub decay: DecayConfig,
    pub log: LogConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbConfig {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedConfig {
    pub provider: String,
    pub model: String,
    pub dims: usize,
    pub recall_model: String,
    pub base_url: String,
    pub api_key_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallConfig {
    pub weights: RecallWeights,
    pub mmr_lambda: f64,
    pub per_source_limit: usize,
    pub adaptive_floor: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallWeights {
    pub vector: f64,
    pub sparse: f64,
    pub colbert: f64,
    pub keyword: f64,
    pub ppr: f64,
    pub tag_overlap: f64,
    pub exact_phrase: f64,
    pub importance: f64,
    pub recency: f64,
    pub confidence: f64,
    pub reliability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PprConfig {
    pub alpha: f32,
    pub epsilon: f32,
    pub max_pushes: usize,
    pub csr_rebuild_threshold: usize,
    pub csr_rebuild_interval_s: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentConfig {
    pub soft_limit_bytes: usize,
    pub hard_limit_bytes: usize,
    pub auto_summarize: bool,
    pub summary_target_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationConfig {
    pub provider: String,
    pub model: String,
    pub cache_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    pub base: f64,
    pub floor_factor: f64,
    pub archive_threshold: f64,
    pub delete_threshold: f64,
    pub grace_days: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    pub level: String,
    pub file: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            db: DbConfig {
                path: default_db_path(),
            },
            embed: EmbedConfig {
                provider: "fastembed-bgem3".to_string(),
                model: "BGEM3Q".to_string(),
                dims: 1024,
                recall_model: String::new(),
                base_url: String::new(),
                api_key_env: "OPENAI_API_KEY".to_string(),
            },
            recall: RecallConfig {
                weights: RecallWeights {
                    vector: 0.20,
                    sparse: 0.15,
                    colbert: 0.20,
                    keyword: 0.15,
                    ppr: 0.10,
                    tag_overlap: 0.05,
                    exact_phrase: 0.03,
                    importance: 0.04,
                    recency: 0.04,
                    confidence: 0.02,
                    reliability: 0.02,
                },
                mmr_lambda: 0.7,
                per_source_limit: 200,
                adaptive_floor: true,
            },
            ppr: PprConfig {
                alpha: 0.15,
                epsilon: 1.0e-4,
                max_pushes: 50_000,
                csr_rebuild_threshold: 1024,
                csr_rebuild_interval_s: 60,
            },
            content: ContentConfig {
                soft_limit_bytes: 500,
                hard_limit_bytes: 2_000,
                auto_summarize: true,
                summary_target_chars: 300,
            },
            classification: ClassificationConfig {
                provider: "local".to_string(),
                model: String::new(),
                cache_size: 4096,
            },
            decay: DecayConfig {
                base: 0.005,
                floor_factor: 0.10,
                archive_threshold: 0.05,
                delete_threshold: 0.01,
                grace_days: 30,
            },
            log: LogConfig {
                level: "info".to_string(),
                file: String::new(),
            },
        }
    }
}

impl Config {
    pub fn load(config_path: Option<PathBuf>, db_path: Option<PathBuf>) -> anyhow::Result<Self> {
        let mut cfg = if let Some(path) = config_path.or_else(default_config_path) {
            if path.exists() {
                let text = fs::read_to_string(&path)
                    .with_context(|| format!("reading {}", path.display()))?;
                merge_defaults(
                    toml::from_str::<PartialConfig>(&text)
                        .with_context(|| format!("parsing {}", path.display()))?,
                )
            } else {
                Self::default()
            }
        } else {
            Self::default()
        };

        if let Ok(path) = env::var("AGSKMEM_DB")
            && !path.trim().is_empty()
        {
            cfg.db.path = expand_home(&path);
        }
        if let Some(path) = db_path {
            cfg.db.path = path;
        }
        if let Ok(level) = env::var("AGSKMEM_LOG_LEVEL") {
            cfg.log.level = level;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.embed.dims == 0 || self.embed.dims > 4096 {
            bail!("embed.dims must be between 1 and 4096");
        }
        match self.embed.provider.trim().to_ascii_lowercase().as_str() {
            "local" | "local-hash" => {}
            "fastembed" | "fastembed-bgem3" | "bge-m3" | "bgem3" => {
                if !matches!(
                    self.embed.model.trim(),
                    "BGEM3Q" | "bge-m3-q" | "gpahal/bge-m3-onnx-int8"
                ) {
                    bail!("unsupported fastembed BGE-M3 model {}", self.embed.model);
                }
                if self.embed.dims != 1024 {
                    bail!("fastembed BGE-M3 requires embed.dims = 1024");
                }
            }
            other => bail!("unsupported embed.provider {other}"),
        }
        let sum = self.recall.weights.vector
            + self.recall.weights.sparse
            + self.recall.weights.colbert
            + self.recall.weights.keyword
            + self.recall.weights.ppr
            + self.recall.weights.tag_overlap
            + self.recall.weights.exact_phrase
            + self.recall.weights.importance
            + self.recall.weights.recency
            + self.recall.weights.confidence
            + self.recall.weights.reliability;
        if (sum - 1.0).abs() > 0.000_001 {
            bail!("recall weights must sum to 1.0, got {sum}");
        }
        if self.content.hard_limit_bytes == 0 {
            bail!("content.hard_limit_bytes must be greater than zero");
        }
        if self.content.soft_limit_bytes > self.content.hard_limit_bytes {
            bail!("content.soft_limit_bytes must be <= content.hard_limit_bytes");
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Default)]
struct PartialConfig {
    db: Option<PartialDbConfig>,
    embed: Option<PartialEmbedConfig>,
    content: Option<PartialContentConfig>,
    classification: Option<PartialClassificationConfig>,
    recall: Option<PartialRecallConfig>,
    ppr: Option<PartialPprConfig>,
    decay: Option<PartialDecayConfig>,
    log: Option<PartialLogConfig>,
}
#[derive(Debug, Deserialize)]
struct PartialDbConfig {
    path: Option<String>,
}
#[derive(Debug, Deserialize)]
struct PartialEmbedConfig {
    provider: Option<String>,
    model: Option<String>,
    dims: Option<usize>,
    recall_model: Option<String>,
    base_url: Option<String>,
    api_key_env: Option<String>,
}
#[derive(Debug, Deserialize)]
struct PartialRecallConfig {
    weights: Option<RecallWeights>,
    mmr_lambda: Option<f64>,
    per_source_limit: Option<usize>,
    adaptive_floor: Option<bool>,
}
#[derive(Debug, Deserialize)]
struct PartialPprConfig {
    alpha: Option<f32>,
    epsilon: Option<f32>,
    max_pushes: Option<usize>,
    csr_rebuild_threshold: Option<usize>,
    csr_rebuild_interval_s: Option<u64>,
}
#[derive(Debug, Deserialize)]
struct PartialDecayConfig {
    base: Option<f64>,
    floor_factor: Option<f64>,
    archive_threshold: Option<f64>,
    delete_threshold: Option<f64>,
    grace_days: Option<i64>,
}
#[derive(Debug, Deserialize)]
struct PartialContentConfig {
    soft_limit_bytes: Option<usize>,
    hard_limit_bytes: Option<usize>,
    auto_summarize: Option<bool>,
    summary_target_chars: Option<usize>,
}
#[derive(Debug, Deserialize)]
struct PartialClassificationConfig {
    provider: Option<String>,
    model: Option<String>,
    cache_size: Option<usize>,
}
#[derive(Debug, Deserialize)]
struct PartialLogConfig {
    level: Option<String>,
    file: Option<String>,
}

fn merge_defaults(partial: PartialConfig) -> Config {
    let mut cfg = Config::default();
    if let Some(db) = partial.db
        && let Some(path) = db.path
    {
        cfg.db.path = expand_home(&path);
    }
    if let Some(embed) = partial.embed {
        if let Some(v) = embed.provider {
            cfg.embed.provider = v;
        }
        if let Some(v) = embed.model {
            cfg.embed.model = v;
        }
        if let Some(v) = embed.dims {
            cfg.embed.dims = v;
        }
        if let Some(v) = embed.recall_model {
            cfg.embed.recall_model = v;
        }
        if let Some(v) = embed.base_url {
            cfg.embed.base_url = v;
        }
        if let Some(v) = embed.api_key_env {
            cfg.embed.api_key_env = v;
        }
    }
    if let Some(recall) = partial.recall {
        if let Some(v) = recall.weights {
            cfg.recall.weights = v;
        }
        if let Some(v) = recall.mmr_lambda {
            cfg.recall.mmr_lambda = v;
        }
        if let Some(v) = recall.per_source_limit {
            cfg.recall.per_source_limit = v;
        }
        if let Some(v) = recall.adaptive_floor {
            cfg.recall.adaptive_floor = v;
        }
    }
    if let Some(ppr) = partial.ppr {
        if let Some(v) = ppr.alpha {
            cfg.ppr.alpha = v;
        }
        if let Some(v) = ppr.epsilon {
            cfg.ppr.epsilon = v;
        }
        if let Some(v) = ppr.max_pushes {
            cfg.ppr.max_pushes = v;
        }
        if let Some(v) = ppr.csr_rebuild_threshold {
            cfg.ppr.csr_rebuild_threshold = v;
        }
        if let Some(v) = ppr.csr_rebuild_interval_s {
            cfg.ppr.csr_rebuild_interval_s = v;
        }
    }
    if let Some(decay) = partial.decay {
        if let Some(v) = decay.base {
            cfg.decay.base = v;
        }
        if let Some(v) = decay.floor_factor {
            cfg.decay.floor_factor = v;
        }
        if let Some(v) = decay.archive_threshold {
            cfg.decay.archive_threshold = v;
        }
        if let Some(v) = decay.delete_threshold {
            cfg.decay.delete_threshold = v;
        }
        if let Some(v) = decay.grace_days {
            cfg.decay.grace_days = v;
        }
    }
    if let Some(content) = partial.content {
        if let Some(v) = content.soft_limit_bytes {
            cfg.content.soft_limit_bytes = v;
        }
        if let Some(v) = content.hard_limit_bytes {
            cfg.content.hard_limit_bytes = v;
        }
        if let Some(v) = content.auto_summarize {
            cfg.content.auto_summarize = v;
        }
        if let Some(v) = content.summary_target_chars {
            cfg.content.summary_target_chars = v;
        }
    }
    if let Some(classification) = partial.classification {
        if let Some(v) = classification.provider {
            cfg.classification.provider = v;
        }
        if let Some(v) = classification.model {
            cfg.classification.model = v;
        }
        if let Some(v) = classification.cache_size {
            cfg.classification.cache_size = v;
        }
    }
    if let Some(log) = partial.log {
        if let Some(v) = log.level {
            cfg.log.level = v;
        }
        if let Some(v) = log.file {
            cfg.log.file = v;
        }
    }
    cfg
}

pub fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|p| p.join("agskmem/config.toml"))
}

pub fn default_db_path() -> PathBuf {
    if let Ok(path) = env::var("AGSKMEM_DB")
        && !path.trim().is_empty()
    {
        return expand_home(&path);
    }
    let base = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(dirs::data_dir)
        .unwrap_or_else(|| expand_home("~/.local/share"));
    base.join("agskmem/agskmem.sqlite3")
}

pub fn expand_home(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(stripped);
    }
    PathBuf::from(path)
}
