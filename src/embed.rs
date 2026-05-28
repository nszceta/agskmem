use sha2::{Digest, Sha256};
use unicode_segmentation::UnicodeSegmentation;

pub trait Embedder: Send + Sync {
    fn model(&self) -> &str;
    fn dims(&self) -> usize;
    fn embed_for_store(&self, texts: &[&str]) -> Vec<Vec<f32>>;
    fn embed_for_recall(&self, texts: &[&str]) -> Vec<Vec<f32>>;
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

    fn embed_for_store(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|text| hash_embed(text, self.dims))
            .collect()
    }

    fn embed_for_recall(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        self.embed_for_store(texts)
    }
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
