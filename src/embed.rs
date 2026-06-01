use anyhow::{Context, bail};
use fastembed::{Bgem3Embedding, Bgem3InitOptions, Bgem3Model, SparseEmbedding};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf, sync::Mutex};
use unicode_segmentation::UnicodeSegmentation;

pub trait Embedder: Send + Sync {
    fn model(&self) -> &str;
    fn dims(&self) -> usize;
    fn embed_for_store(&self, texts: &[&str]) -> anyhow::Result<EmbeddingBatch>;
    fn embed_for_recall(&self, texts: &[&str]) -> anyhow::Result<EmbeddingBatch>;
}

#[derive(Debug, Clone, Default)]
pub struct EmbeddingBatch {
    pub dense: Vec<Vec<f32>>,
    pub sparse: Vec<SparseVector>,
    pub colbert: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone, Default)]
pub struct SparseVector {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LocalHashEmbedder {
    model: String,
    dims: usize,
}

impl LocalHashEmbedder {
    pub fn new(model: String, dims: usize) -> Self {
        Self { model, dims }
    }
}

impl Embedder for LocalHashEmbedder {
    fn model(&self) -> &str {
        &self.model
    }
    fn dims(&self) -> usize {
        self.dims
    }

    fn embed_for_store(&self, texts: &[&str]) -> anyhow::Result<EmbeddingBatch> {
        let mut dense = Vec::with_capacity(texts.len());
        let mut sparse = Vec::with_capacity(texts.len());
        let mut colbert = Vec::with_capacity(texts.len());
        for text in texts {
            dense.push(hash_embed(text, self.dims));
            sparse.push(local_sparse(text));
            colbert.push(local_colbert(text, self.dims));
        }
        Ok(EmbeddingBatch {
            dense,
            sparse,
            colbert,
        })
    }

    fn embed_for_recall(&self, texts: &[&str]) -> anyhow::Result<EmbeddingBatch> {
        self.embed_for_store(texts)
    }
}

pub struct FastEmbedBgeM3Embedder {
    model: String,
    dims: usize,
    bgem3_model: Bgem3Model,
    cache_dir: PathBuf,
    inner: Mutex<Option<Bgem3Embedding>>,
}

impl FastEmbedBgeM3Embedder {
    pub fn new(model: String, dims: usize, cache_dir: PathBuf) -> anyhow::Result<Self> {
        let bgem3_model = parse_bgem3_model(&model)?;
        Ok(Self {
            model,
            dims,
            bgem3_model,
            cache_dir,
            inner: Mutex::new(None),
        })
    }

    fn init_options(&self) -> Bgem3InitOptions {
        Bgem3InitOptions::new(self.bgem3_model.clone()).with_cache_dir(self.cache_dir.clone())
    }
}

impl Embedder for FastEmbedBgeM3Embedder {
    fn model(&self) -> &str {
        &self.model
    }

    fn dims(&self) -> usize {
        self.dims
    }
    fn embed_for_store(&self, texts: &[&str]) -> anyhow::Result<EmbeddingBatch> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("fastembed BGE-M3 embedder mutex poisoned"))?;
        if guard.is_none() {
            *guard = Some(Bgem3Embedding::try_new(self.init_options())?);
        }
        let inner = guard
            .as_mut()
            .context("fastembed BGE-M3 embedder was not initialized")?;
        let output = inner.embed(texts, Some(8))?;
        validate_dense_dims(&output.dense, self.dims)?;
        Ok(EmbeddingBatch {
            dense: output.dense,
            sparse: output
                .sparse
                .into_iter()
                .map(sparse_from_fastembed)
                .collect(),
            colbert: output.colbert,
        })
    }

    fn embed_for_recall(&self, texts: &[&str]) -> anyhow::Result<EmbeddingBatch> {
        self.embed_for_store(texts)
    }
}

fn parse_bgem3_model(model: &str) -> anyhow::Result<Bgem3Model> {
    match model.trim() {
        "BGEM3Q" | "bge-m3-q" | "gpahal/bge-m3-onnx-int8" => Ok(Bgem3Model::BGEM3Q),
        other => bail!("unsupported fastembed BGE-M3 model {other}"),
    }
}

fn validate_dense_dims(vectors: &[Vec<f32>], expected: usize) -> anyhow::Result<()> {
    for (i, vector) in vectors.iter().enumerate() {
        if vector.len() != expected {
            bail!(
                "fastembed BGE-M3 returned {} dims for vector {i}, expected {expected}",
                vector.len()
            );
        }
    }
    Ok(())
}

fn sparse_from_fastembed(value: SparseEmbedding) -> SparseVector {
    SparseVector {
        indices: value.indices,
        values: value.values,
    }
}

fn local_sparse(text: &str) -> SparseVector {
    let mut weights = BTreeMap::<usize, f32>::new();
    for token in text.unicode_words().map(|w| w.to_ascii_lowercase()) {
        if token.is_empty() {
            continue;
        }
        *weights.entry(sparse_token_id(&token)).or_default() += 1.0;
    }
    let mut indices = Vec::with_capacity(weights.len());
    let mut values = Vec::with_capacity(weights.len());
    for (index, value) in weights {
        indices.push(index);
        values.push(value);
    }
    SparseVector { indices, values }
}

fn local_colbert(text: &str, dims: usize) -> Vec<Vec<f32>> {
    text.unicode_words()
        .map(|w| w.to_ascii_lowercase())
        .filter(|token| !token.is_empty())
        .map(|token| hash_embed(&token, dims))
        .collect()
}

fn sparse_token_id(token: &str) -> usize {
    let digest = Sha256::digest(token.as_bytes());
    (u64::from_le_bytes(digest[0..8].try_into().expect("sha256 has 32 bytes"))
        & 0x7fff_ffff_ffff_ffff) as usize
}

pub fn hash_embed(text: &str, dims: usize) -> Vec<f32> {
    let mut vec = vec![0.0_f32; dims];
    for token in text.unicode_words().map(|w| w.to_ascii_lowercase()) {
        if token.is_empty() {
            continue;
        }
        let digest = Sha256::digest(token.as_bytes());
        let bucket = u64::from_le_bytes(digest[0..8].try_into().expect("sha256 has 32 bytes"))
            as usize
            % dims;
        let sign = if digest[8] & 1 == 0 { 1.0 } else { -1.0 };
        vec[bucket] += sign;
    }
    normalize(&mut vec);
    vec
}

pub fn normalize(vec: &mut [f32]) -> f32 {
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
    norm
}

pub fn vector_to_blob(vec: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vec.len() * 4);
    for value in vec {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

pub fn blob_to_vector(blob: &[u8]) -> anyhow::Result<Vec<f32>> {
    if !blob.len().is_multiple_of(4) {
        anyhow::bail!("embedding blob length {} is not divisible by 4", blob.len());
    }
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        out.push(f32::from_le_bytes(
            chunk.try_into().expect("chunk_exact yielded four bytes"),
        ));
    }
    Ok(out)
}

pub fn cosine_from_blobs(a: &[u8], b: &[u8]) -> anyhow::Result<f64> {
    let a = blob_to_vector(a)?;
    let b = blob_to_vector(b)?;
    if a.len() != b.len() {
        anyhow::bail!("embedding dimensions differ: {} != {}", a.len(), b.len());
    }
    Ok(cosine(&a, &b) as f64)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    dot.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastembed_bgem3_init_uses_configured_cache_dir() {
        let cache_dir = PathBuf::from("/tmp/agskmem-fastembed-cache-test");
        let embedder = FastEmbedBgeM3Embedder::new("BGEM3Q".to_string(), 1024, cache_dir.clone())
            .expect("valid BGE-M3 embedder");

        assert_eq!(embedder.init_options().cache_dir, cache_dir);
    }
}
