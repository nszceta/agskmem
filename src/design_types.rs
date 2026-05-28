use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StoreMemoryArgs {
    pub content: Option<String>,
    #[serde(default)]
    pub memories: Vec<StoreOneArgs>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub importance: Option<f64>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default, rename = "type")]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub t_valid: Option<String>,
    #[serde(default)]
    pub t_invalid: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StoreOneArgs {
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub importance: Option<f64>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default, rename = "type")]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub t_valid: Option<String>,
    #[serde(default)]
    pub t_invalid: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpdateMemoryArgs {
    pub memory_id: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default, rename = "type")]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub importance: Option<f64>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub relevance: Option<f64>,
    #[serde(default)]
    pub reliability: Option<f64>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub t_valid: Option<String>,
    #[serde(default)]
    pub t_invalid: Option<String>,
    #[serde(default)]
    pub archived: Option<bool>,
    #[serde(default)]
    pub protected: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeleteMemoryArgs {
    #[serde(default)]
    pub memory_id: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub confirmation_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociateArgs {
    pub memory1_id: String,
    pub memory2_id: String,
    #[serde(rename = "type")]
    pub relation_type: String,
    #[serde(default)]
    pub strength: Option<f64>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecallArgs {
    #[serde(default)]
    pub memory_id: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub queries: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub exclude_tags: Vec<String>,
    #[serde(default)]
    pub context_tags: Vec<String>,
    #[serde(default)]
    pub context: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub active_path: Option<String>,
    #[serde(default)]
    pub context_types: Vec<String>,
    #[serde(default)]
    pub tag_mode: Option<String>,
    #[serde(default)]
    pub tag_match: Option<String>,
    #[serde(default)]
    pub priority_ids: Vec<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub cursor: Option<usize>,
    #[serde(default)]
    pub current_only: Option<bool>,
    #[serde(default)]
    pub state_debug: bool,
    #[serde(default)]
    pub expand_relations: bool,
    #[serde(default)]
    pub expand_entities: bool,
    #[serde(default)]
    pub expand_respect_tags: bool,
    #[serde(default)]
    pub time_query: Option<String>,
    #[serde(default)]
    pub sort: Option<String>,
    #[serde(default)]
    pub order_by: Option<String>,
    #[serde(default)]
    pub relation_limit: Option<usize>,
    #[serde(default)]
    pub expansion_limit: Option<usize>,
    #[serde(default)]
    pub expand_min_importance: Option<f64>,
    #[serde(default)]
    pub expand_min_strength: Option<f64>,
    #[serde(default)]
    pub auto_decompose: bool,
    #[serde(default)]
    pub min_score: Option<f64>,
    #[serde(default)]
    pub adaptive_floor: Option<bool>,
    #[serde(default)]
    pub as_of: Option<String>,
    #[serde(default)]
    pub start: Option<String>,
    #[serde(default)]
    pub end: Option<String>,
    #[serde(default)]
    pub time_range: Option<TimeRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimeRange {
    #[serde(default)]
    pub start: Option<String>,
    #[serde(default)]
    pub end: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RelatedArgs {
    pub memory_id: String,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphNeighborsArgs {
    pub memory_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsolidateArgs {
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReembedArgs {
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub memory_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnrichmentReprocessArgs {
    #[serde(default)]
    pub ids: Vec<String>,
    #[serde(default)]
    pub forced: bool,
}
