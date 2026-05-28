use crate::model::RelationKind;
use arc_swap::ArcSwap;
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub id_to_node: HashMap<String, NodeId>,
    pub node_to_id: Vec<String>,
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<NodeId>,
    pub kind: Vec<u8>,
    pub strength: Vec<f32>,
    pub confidence: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphStats {
    pub nodes: usize,
    pub directed_edges: usize,
    pub max_out_degree: usize,
}

impl Graph {
    pub fn from_edges(edges: Vec<(String, String, RelationKind, f32, f32)>) -> Self {
        let mut ids: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for (src, dst, _, _, _) in &edges {
            if seen.insert(src.clone()) {
                ids.push(src.clone());
            }
            if seen.insert(dst.clone()) {
                ids.push(dst.clone());
            }
        }
        ids.sort_unstable();
        let id_to_node: HashMap<String, NodeId> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), NodeId(i as u32)))
            .collect();
        let mut adjacency: Vec<Vec<(NodeId, RelationKind, f32, f32)>> = vec![Vec::new(); ids.len()];
        for (src, dst, kind, strength, confidence) in edges {
            if let (Some(s), Some(d)) = (id_to_node.get(&src), id_to_node.get(&dst)) {
                adjacency[s.0 as usize].push((*d, kind, strength, confidence));
            }
        }
        for edges in &mut adjacency {
            edges.sort_by_key(|(dst, kind, _, _)| (dst.0, *kind as u8));
        }
        let mut row_ptr = Vec::with_capacity(ids.len() + 1);
        let mut col_idx = Vec::new();
        let mut kind = Vec::new();
        let mut strength = Vec::new();
        let mut confidence = Vec::new();
        row_ptr.push(0);
        for list in adjacency {
            for (dst, rel, edge_strength, edge_confidence) in list {
                col_idx.push(dst);
                kind.push(rel as u8);
                strength.push(edge_strength);
                confidence.push(edge_confidence);
            }
            row_ptr.push(col_idx.len() as u32);
        }
        Self {
            id_to_node,
            node_to_id: ids,
            row_ptr,
            col_idx,
            kind,
            strength,
            confidence,
        }
    }

    pub fn stats(&self) -> GraphStats {
        let max_out_degree = self
            .row_ptr
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        GraphStats {
            nodes: self.node_to_id.len(),
            directed_edges: self.col_idx.len(),
            max_out_degree,
        }
    }

    pub fn out_degree(&self, node: NodeId) -> usize {
        let i = node.0 as usize;
        if i + 1 >= self.row_ptr.len() {
            return 0;
        }
        (self.row_ptr[i + 1] - self.row_ptr[i]) as usize
    }

    pub fn neighbors(&self, id: &str) -> Vec<Neighbor> {
        let Some(node) = self.id_to_node.get(id).copied() else {
            return Vec::new();
        };
        let start = self.row_ptr[node.0 as usize] as usize;
        let end = self.row_ptr[node.0 as usize + 1] as usize;
        (start..end)
            .map(|i| Neighbor {
                id: self.node_to_id[self.col_idx[i].0 as usize].clone(),
                kind: RelationKind::from_i64(i64::from(self.kind[i]))
                    .map(|k| k.as_str().to_string())
                    .unwrap_or_else(|_| "UNKNOWN".to_string()),
                strength: self.strength[i],
                confidence: self.confidence[i],
            })
            .collect()
    }

    pub fn forward_push(
        &self,
        seed_scores: &HashMap<String, f32>,
        alpha: f32,
        epsilon: f32,
        max_pushes: usize,
    ) -> HashMap<String, f32> {
        if self.node_to_id.is_empty() {
            return HashMap::new();
        }
        let mut residual: HashMap<NodeId, f32> = HashMap::new();
        let mut total = 0.0_f32;
        for (id, score) in seed_scores {
            if *score <= 0.0 {
                continue;
            }
            if let Some(node) = self.id_to_node.get(id) {
                total += *score;
                *residual.entry(*node).or_default() += *score;
            }
        }
        if total <= 0.0 {
            return HashMap::new();
        }
        for value in residual.values_mut() {
            *value /= total;
        }

        let threshold = epsilon / self.node_to_id.len() as f32;
        let mut rank: HashMap<NodeId, f32> = HashMap::new();
        let mut frontier = VecDeque::new();
        let mut in_frontier = HashSet::new();
        for (&node, &value) in &residual {
            if value >= threshold * self.out_degree(node).max(1) as f32 {
                frontier.push_back(node);
                in_frontier.insert(node);
            }
        }
        let mut pushes = 0;
        while let Some(node) = frontier.pop_front() {
            in_frontier.remove(&node);
            if pushes >= max_pushes {
                break;
            }
            let ru = residual.remove(&node).unwrap_or(0.0);
            let deg = self.out_degree(node);
            if ru < threshold * deg.max(1) as f32 {
                continue;
            }
            *rank.entry(node).or_default() += alpha * ru;
            let push = (1.0 - alpha) * ru;
            if deg == 0 {
                *rank.entry(node).or_default() += push;
                pushes += 1;
                continue;
            }
            let start = self.row_ptr[node.0 as usize] as usize;
            let end = self.row_ptr[node.0 as usize + 1] as usize;
            let denom = (start..end)
                .map(|i| edge_weight(self.kind[i], self.strength[i], self.confidence[i]))
                .sum::<f32>();
            if denom <= 0.0 {
                continue;
            }
            for i in start..end {
                let dst = self.col_idx[i];
                let contribution =
                    push * edge_weight(self.kind[i], self.strength[i], self.confidence[i]) / denom;
                let entry = residual.entry(dst).or_default();
                *entry += contribution;
                if *entry >= threshold * self.out_degree(dst).max(1) as f32
                    && in_frontier.insert(dst)
                {
                    frontier.push_back(dst);
                }
            }
            pushes += 1;
        }
        let norm = rank.values().sum::<f32>();
        if norm > 0.0 {
            rank.into_iter()
                .map(|(node, score)| (self.node_to_id[node.0 as usize].clone(), score / norm))
                .collect()
        } else {
            HashMap::new()
        }
    }
}

fn edge_weight(kind: u8, strength: f32, confidence: f32) -> f32 {
    let relation = RelationKind::from_i64(i64::from(kind))
        .map(RelationKind::default_weight)
        .unwrap_or(0.1);
    relation * strength.clamp(0.0, 1.0) * confidence.clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Serialize)]
pub struct Neighbor {
    pub id: String,
    pub kind: String,
    pub strength: f32,
    pub confidence: f32,
}

#[derive(Debug)]
pub struct GraphStore {
    inner: ArcSwap<Graph>,
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new(Graph::default())
    }
}

impl GraphStore {
    pub fn new(graph: Graph) -> Self {
        Self {
            inner: ArcSwap::from_pointee(graph),
        }
    }
    pub fn load(&self) -> Arc<Graph> {
        self.inner.load_full()
    }
    pub fn publish(&self, graph: Graph) {
        self.inner.store(Arc::new(graph));
    }
}
