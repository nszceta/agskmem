use anyhow::{anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

pub const MAX_CONTENT_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    Decision = 1,
    Pattern = 2,
    Preference = 3,
    Style = 4,
    Habit = 5,
    Insight = 6,
    Context = 7,
    Statement = 8,
}

impl MemoryType {
    pub fn from_i64(value: i64) -> anyhow::Result<Self> {
        match value {
            1 => Ok(Self::Decision),
            2 => Ok(Self::Pattern),
            3 => Ok(Self::Preference),
            4 => Ok(Self::Style),
            5 => Ok(Self::Habit),
            6 => Ok(Self::Insight),
            7 => Ok(Self::Context),
            8 => Ok(Self::Statement),
            _ => bail!("unknown memory type code {value}"),
        }
    }

    pub fn parse(value: Option<&str>) -> anyhow::Result<Self> {
        let Some(value) = value else {
            return Ok(Self::Context);
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "decision" => Ok(Self::Decision),
            "pattern" => Ok(Self::Pattern),
            "preference" => Ok(Self::Preference),
            "style" => Ok(Self::Style),
            "habit" => Ok(Self::Habit),
            "insight" => Ok(Self::Insight),
            "context" | "memory" => Ok(Self::Context),
            "statement" => Ok(Self::Statement),
            other => bail!("unknown memory type {other}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Decision => "Decision",
            Self::Pattern => "Pattern",
            Self::Preference => "Preference",
            Self::Style => "Style",
            Self::Habit => "Habit",
            Self::Insight => "Insight",
            Self::Context => "Context",
            Self::Statement => "Statement",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationKind {
    RelatesTo = 1,
    LeadsTo = 2,
    OccurredBefore = 3,
    PrefersOver = 4,
    Exemplifies = 5,
    Contradicts = 6,
    Reinforces = 7,
    InvalidatedBy = 8,
    EvolvedInto = 9,
    DerivedFrom = 10,
    PartOf = 11,
    SimilarTo = 12,
    PrecededBy = 13,
    Discovered = 14,
    ExtractedFrom = 15,
}

impl RelationKind {
    pub fn parse_authorable(value: &str) -> anyhow::Result<Self> {
        let kind = Self::parse(value)?;
        if kind.is_authorable() {
            Ok(kind)
        } else {
            bail!("relation kind {} is system-managed", kind.as_str())
        }
    }

    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value.trim().to_ascii_uppercase().as_str() {
            "RELATES_TO" => Ok(Self::RelatesTo),
            "LEADS_TO" => Ok(Self::LeadsTo),
            "OCCURRED_BEFORE" => Ok(Self::OccurredBefore),
            "PREFERS_OVER" => Ok(Self::PrefersOver),
            "EXEMPLIFIES" => Ok(Self::Exemplifies),
            "CONTRADICTS" => Ok(Self::Contradicts),
            "REINFORCES" => Ok(Self::Reinforces),
            "INVALIDATED_BY" => Ok(Self::InvalidatedBy),
            "EVOLVED_INTO" => Ok(Self::EvolvedInto),
            "DERIVED_FROM" => Ok(Self::DerivedFrom),
            "PART_OF" => Ok(Self::PartOf),
            "SIMILAR_TO" => Ok(Self::SimilarTo),
            "PRECEDED_BY" => Ok(Self::PrecededBy),
            "DISCOVERED" => Ok(Self::Discovered),
            "EXTRACTED_FROM" => Ok(Self::ExtractedFrom),
            other => bail!("unknown relation kind {other}"),
        }
    }

    pub fn from_i64(value: i64) -> anyhow::Result<Self> {
        match value {
            1 => Ok(Self::RelatesTo),
            2 => Ok(Self::LeadsTo),
            3 => Ok(Self::OccurredBefore),
            4 => Ok(Self::PrefersOver),
            5 => Ok(Self::Exemplifies),
            6 => Ok(Self::Contradicts),
            7 => Ok(Self::Reinforces),
            8 => Ok(Self::InvalidatedBy),
            9 => Ok(Self::EvolvedInto),
            10 => Ok(Self::DerivedFrom),
            11 => Ok(Self::PartOf),
            12 => Ok(Self::SimilarTo),
            13 => Ok(Self::PrecededBy),
            14 => Ok(Self::Discovered),
            15 => Ok(Self::ExtractedFrom),
            _ => bail!("unknown relation kind code {value}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RelatesTo => "RELATES_TO",
            Self::LeadsTo => "LEADS_TO",
            Self::OccurredBefore => "OCCURRED_BEFORE",
            Self::PrefersOver => "PREFERS_OVER",
            Self::Exemplifies => "EXEMPLIFIES",
            Self::Contradicts => "CONTRADICTS",
            Self::Reinforces => "REINFORCES",
            Self::InvalidatedBy => "INVALIDATED_BY",
            Self::EvolvedInto => "EVOLVED_INTO",
            Self::DerivedFrom => "DERIVED_FROM",
            Self::PartOf => "PART_OF",
            Self::SimilarTo => "SIMILAR_TO",
            Self::PrecededBy => "PRECEDED_BY",
            Self::Discovered => "DISCOVERED",
            Self::ExtractedFrom => "EXTRACTED_FROM",
        }
    }

    pub fn is_authorable(self) -> bool {
        matches!(
            self,
            Self::RelatesTo
                | Self::LeadsTo
                | Self::OccurredBefore
                | Self::PrefersOver
                | Self::Exemplifies
                | Self::Contradicts
                | Self::Reinforces
                | Self::InvalidatedBy
                | Self::EvolvedInto
                | Self::DerivedFrom
                | Self::PartOf
        )
    }

    pub fn default_weight(self) -> f32 {
        match self {
            Self::Exemplifies => 1.00,
            Self::DerivedFrom => 0.90,
            Self::LeadsTo | Self::Reinforces => 0.80,
            Self::PartOf | Self::EvolvedInto => 0.75,
            Self::ExtractedFrom => 0.70,
            Self::PrefersOver => 0.60,
            Self::SimilarTo => 0.55,
            Self::RelatesTo => 0.50,
            Self::Discovered => 0.45,
            Self::OccurredBefore => 0.40,
            Self::PrecededBy => 0.35,
            Self::Contradicts => 0.30,
            Self::InvalidatedBy => 0.20,
        }
    }
}

impl fmt::Display for RelationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRow {
    pub id: String,
    pub content: String,
    pub summary: Option<String>,
    #[serde(rename = "type")]
    pub memory_type: String,
    pub importance: f64,
    pub confidence: f64,
    pub relevance: f64,
    pub reliability: f64,
    pub metadata: Value,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
    pub last_accessed: Option<String>,
    pub t_valid: Option<String>,
    pub t_invalid: Option<String>,
    pub archived: bool,
    pub protected: bool,
    pub state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScoreComponents {
    pub vector: f64,
    pub keyword: f64,
    pub ppr: f64,
    pub tag_overlap: f64,
    pub exact_phrase: f64,
    pub importance: f64,
    pub recency: f64,
    pub confidence: f64,
    pub reliability: f64,
    pub context_bonus: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallHit {
    #[serde(flatten)]
    pub memory: MemoryRow,
    pub score: f64,
    pub components: ScoreComponents,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub state_replacements: Vec<String>,
}

pub fn now_epoch() -> i64 {
    OffsetDateTime::now_utc().unix_timestamp()
}

pub fn epoch_to_rfc3339(epoch: i64) -> anyhow::Result<String> {
    OffsetDateTime::from_unix_timestamp(epoch)
        .map_err(|e| anyhow!(e))?
        .format(&Rfc3339)
        .map_err(|e| anyhow!(e))
}

pub fn opt_epoch_to_rfc3339(epoch: Option<i64>) -> anyhow::Result<Option<String>> {
    epoch.map(epoch_to_rfc3339).transpose()
}

pub fn parse_time(value: Option<&str>) -> anyhow::Result<Option<i64>> {
    match value {
        None | Some("") => Ok(None),
        Some(raw) => {
            if let Ok(epoch) = raw.parse::<i64>() {
                return Ok(Some(epoch));
            }
            let dt = OffsetDateTime::parse(raw, &Rfc3339)
                .map_err(|e| anyhow!("invalid RFC3339 timestamp {raw}: {e}"))?;
            Ok(Some(dt.unix_timestamp()))
        }
    }
}

pub fn normalize_tags(values: &[String]) -> Vec<String> {
    let mut out: Vec<String> = values
        .iter()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

pub fn json_object_or_empty(value: Option<Value>) -> anyhow::Result<String> {
    match value.unwrap_or(Value::Object(Map::new())) {
        Value::Object(map) => serde_json::to_string(&Value::Object(map)).map_err(Into::into),
        _ => bail!("metadata must be a JSON object"),
    }
}

pub fn clamp_unit(value: Option<f64>, default: f64, field: &str) -> anyhow::Result<f64> {
    let value = value.unwrap_or(default);
    if !(0.0..=1.0).contains(&value) || value.is_nan() {
        bail!("{field} must be in [0, 1]");
    }
    Ok(value)
}

pub fn validate_content(content: &str, hard_limit_bytes: usize) -> anyhow::Result<()> {
    let len = content.len();
    if len == 0 {
        bail!("content must not be empty");
    }
    if len > hard_limit_bytes {
        bail!(
            "content is {len} bytes, exceeding the configured {hard_limit_bytes} byte hard limit"
        );
    }
    Ok(())
}
