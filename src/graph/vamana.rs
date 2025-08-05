//! Vamana graph implementation for DiskANN
//!
//! This module implements the core Vamana algorithm for graph construction
//! and maintenance, following the DiskANN paper.

use crate::{Distance, DistanceFunction, Result, Error};
use crate::distance::create_distance_function;
use parking_lot::RwLock;
use std::collections::BinaryHeap;
use hashbrown::HashSet;
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use serde::{Serialize, Deserialize, Serializer, Deserializer};

/// Neighbor of a vertex in the graph
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Neighbor {
    pub id: usize,
    pub distance: f32,
}

impl Eq for Neighbor {}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Serializable representation of VamanaGraph
#[derive(Serialize, Deserialize)]
pub struct SerializableVamanaGraph {
    graph: Vec<Vec<usize>>,
    max_degree: usize,
    search_list_size: usize,
    alpha: f32,
    num_vertices: usize,
    entry_point: usize,
    metric: Distance,
    dimension: usize,
}

/// Vamana graph structure
pub struct VamanaGraph {
    /// Adjacency list representation
    graph: Arc<RwLock<Vec<Vec<usize>>>>,
    /// Maximum degree per vertex
    max_degree: usize,
    /// Search list size for construction
    search_list_size: usize,
    /// Alpha parameter for pruning
    alpha: f32,
    /// Number of vertices
    num_vertices: usize,
    /// Entry point for search
    entry_point: AtomicUsize,
    /// Distance calculator
    distance_fn: Box<dyn DistanceFunction>,
    /// Metric for serialization
    metric: Distance,
    /// Dimension for serialization
    dimension: usize,
}

impl VamanaGraph {
    /// Create a new Vamana graph
    pub fn new(
        num_vertices: usize,
        dimension: usize,
        metric: Distance,
        max_degree: usize,
        search_list_size: usize,
        alpha: f32,
    ) -> Self {
        Self {
            graph: Arc::new(RwLock::new(vec![Vec::new(); num_vertices])),
            max_degree,
            search_list_size,
            alpha,
            num_vertices,
            entry_point: AtomicUsize::new(0),
            distance_fn: create_distance_function(metric, dimension),
            metric,
            dimension,
        }
    }
    
    /// Build the graph from vectors
    pub fn build(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::InvalidParameter("No vectors provided".to_string()).into());
        }
        
        // Find medoid as entry point
        let medoid = self.find_medoid(vectors)?;
        self.entry_point.store(medoid, AtomicOrdering::Relaxed);
        
        // Build graph using Vamana algorithm
        for i in 0..self.num_vertices {
            self.insert_vertex(i, &vectors[i], vectors)?;
        }
        
        // Prune excess edges
        self.prune_graph(vectors)?;
        
        Ok(())
    }
    
    /// Find the medoid (most central point) as entry point
    fn find_medoid(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        let mut min_total_distance = f32::MAX;
        let mut medoid = 0;
        
        // Sample points for efficiency
        let sample_size = (self.num_vertices as f32).sqrt() as usize;
        let step = self.num_vertices / sample_size.max(1);
        
        for i in (0..self.num_vertices).step_by(step.max(1)) {
            let mut total_distance = 0.0;
            
            for j in (0..self.num_vertices).step_by(step.max(1)) {
                if i != j {
                    total_distance += self.distance_fn.distance(&vectors[i], &vectors[j])?;
                }
            }
            
            if total_distance < min_total_distance {
                min_total_distance = total_distance;
                medoid = i;
            }
        }
        
        Ok(medoid)
    }
    
    /// Insert a vertex into the graph
    fn insert_vertex(&self, vertex_id: usize, vertex: &[f32], vectors: &[Vec<f32>]) -> Result<()> {
        // Search for nearest neighbors
        let candidates = self.greedy_search(vertex, self.search_list_size, vectors)?;
        
        // Add bidirectional edges
        let mut graph = self.graph.write();
        
        for &neighbor_id in &candidates {
            if neighbor_id != vertex_id {
                // Add edge from vertex to neighbor
                if !graph[vertex_id].contains(&neighbor_id) {
                    graph[vertex_id].push(neighbor_id);
                }
                
                // Add edge from neighbor to vertex (bidirectional)
                if !graph[neighbor_id].contains(&vertex_id) {
                    graph[neighbor_id].push(vertex_id);
                }
            }
        }
        
        drop(graph);
        
        // Prune if degree exceeds limit
        if candidates.len() > self.max_degree {
            self.prune_vertex(vertex_id, vectors)?;
        }
        
        Ok(())
    }
    
    /// Greedy search for nearest neighbors
    fn greedy_search(&self, query: &[f32], k: usize, vectors: &[Vec<f32>]) -> Result<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        
        // Start from entry point
        let entry = self.entry_point.load(AtomicOrdering::Relaxed);
        let entry_dist = self.distance_fn.distance(query, &vectors[entry])?;
        candidates.push(Neighbor { id: entry, distance: entry_dist });
        w.push(Neighbor { id: entry, distance: entry_dist });
        visited.insert(entry);
        
        while !candidates.is_empty() {
            let current = candidates.pop().unwrap();
            
            // Check termination condition
            if current.distance > w.peek().unwrap().distance * self.alpha {
                break;
            }
            
            // Explore neighbors
            let graph = self.graph.read();
            let neighbors = graph[current.id].clone();
            drop(graph);
            
            for &neighbor_id in &neighbors {
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id);
                    
                    let dist = self.distance_fn.distance(query, &vectors[neighbor_id])?;
                    let neighbor = Neighbor { id: neighbor_id, distance: dist };
                    
                    if dist < w.peek().unwrap().distance || w.len() < k {
                        candidates.push(neighbor);
                        w.push(neighbor);
                        
                        // Maintain size k
                        if w.len() > k {
                            w.pop();
                        }
                    }
                }
            }
        }
        
        // Extract result
        let mut result: Vec<usize> = w.into_iter().map(|n| n.id).collect();
        result.truncate(k);
        Ok(result)
    }
    
    /// Prune edges for a vertex using RobustPrune algorithm
    fn prune_vertex(&self, vertex_id: usize, vectors: &[Vec<f32>]) -> Result<()> {
        let graph = self.graph.read();
        let candidates: Vec<usize> = graph[vertex_id].clone();
        drop(graph);
        
        if candidates.len() <= self.max_degree {
            return Ok(());
        }
        
        // Calculate distances to all candidates
        let mut neighbors: Vec<Neighbor> = candidates
            .iter()
            .map(|&id| {
                let dist = self.distance_fn.distance(
                    &vectors[vertex_id][..],
                    &vectors[id][..]
                ).unwrap();
                Neighbor { id, distance: dist }
            })
            .collect();
        
        // Sort by distance
        neighbors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        // RobustPrune algorithm
        let mut pruned = Vec::new();
        let mut pruned_set = HashSet::new();
        
        for neighbor in neighbors {
            if pruned.len() >= self.max_degree {
                break;
            }
            
            // Check if this neighbor is closer to vertex than to any selected neighbor
            let mut should_prune = false;
            
            for &selected_id in &pruned {
                let dist_to_selected = self.distance_fn.distance(
                    vectors[neighbor.id].as_slice(), 
                    vectors.get(selected_id).map(|v: &Vec<f32>| v.as_slice()).unwrap()
                )?;
                if dist_to_selected < neighbor.distance {
                    should_prune = true;
                    break;
                }
            }
            
            if !should_prune {
                pruned.push(neighbor.id);
                pruned_set.insert(neighbor.id);
            }
        }
        
        // Update graph
        let mut graph = self.graph.write();
        graph[vertex_id] = pruned;
        
        // Remove reverse edges for pruned neighbors
        for &id in &candidates {
            if !pruned_set.contains(&id) {
                graph[id].retain(|&x| x != vertex_id);
            }
        }
        
        Ok(())
    }
    
    /// Prune the entire graph
    fn prune_graph(&self, vectors: &[Vec<f32>]) -> Result<()> {
        for i in 0..self.num_vertices {
            self.prune_vertex(i, vectors)?;
        }
        Ok(())
    }
    
    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, vectors: &[Vec<f32>]) -> Result<Vec<(usize, f32)>> {
        let neighbors = self.greedy_search(query, k, vectors)?;
        
        // Calculate distances for results
        let mut results = Vec::new();
        for &id in &neighbors {
            let dist = self.distance_fn.distance(query, &vectors[id])?;
            results.push((id, dist));
        }
        
        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        
        Ok(results)
    }
    
    /// Search for k nearest neighbors with Option vectors (for dynamic index)
    pub fn search_dynamic(&self, query: &[f32], k: usize, vectors: &[Option<Vec<f32>>]) -> Result<Vec<(usize, f32)>> {
        let neighbors = self.greedy_search_dynamic(query, k, vectors)?;
        
        // Calculate distances for results
        let mut results = Vec::new();
        for &id in &neighbors {
            if let Some(vec) = &vectors[id] {
                let dist = self.distance_fn.distance(query, vec)?;
                results.push((id, dist));
            }
        }
        
        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        
        Ok(results)
    }
    
    /// Get entry point for search
    pub fn get_entry_point(&self) -> usize {
        self.entry_point.load(AtomicOrdering::Relaxed)
    }
    
    /// Get neighbors of a vertex
    pub fn get_neighbors(&self, vertex_id: usize) -> Vec<usize> {
        let graph = self.graph.read();
        if vertex_id < graph.len() {
            graph[vertex_id].clone()
        } else {
            Vec::new()
        }
    }
    
    /// Find a valid entry point for dynamic index
    pub fn find_valid_entry_point(&self, vectors: &[Option<Vec<f32>>]) -> usize {
        let entry = self.entry_point.load(AtomicOrdering::Relaxed);
        if entry < vectors.len() && vectors[entry].is_some() {
            return entry;
        }
        
        // Find first valid vector
        for (i, v) in vectors.iter().enumerate() {
            if v.is_some() {
                return i;
            }
        }
        
        0 // Fallback
    }
    
    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let graph = self.graph.read();
        
        let mut total_degree = 0;
        let mut max_degree = 0;
        let mut min_degree = usize::MAX;
        
        for neighbors in graph.iter() {
            let degree = neighbors.len();
            total_degree += degree;
            max_degree = max_degree.max(degree);
            min_degree = min_degree.min(degree);
        }
        
        GraphStats {
            num_vertices: self.num_vertices,
            num_edges: total_degree,
            avg_degree: total_degree as f32 / self.num_vertices as f32,
            max_degree,
            min_degree,
            entry_point: self.entry_point.load(AtomicOrdering::Relaxed),
        }
    }
    
    /// Insert a single vertex dynamically
    pub fn insert_single(&self, vertex_id: usize, vertex: &[f32], vectors: &[Option<Vec<f32>>]) -> Result<()> {
        // Ensure graph has capacity
        {
            let mut graph = self.graph.write();
            if vertex_id >= graph.len() {
                graph.resize(vertex_id + 1, Vec::new());
            }
        }
        
        // Search for nearest neighbors among existing vertices
        let candidates = self.greedy_search_dynamic(vertex, self.search_list_size, vectors)?;
        
        // Add bidirectional edges
        let mut graph = self.graph.write();
        
        for &neighbor_id in &candidates {
            if neighbor_id != vertex_id {
                // Add edge from vertex to neighbor
                if !graph[vertex_id].contains(&neighbor_id) {
                    graph[vertex_id].push(neighbor_id);
                }
                
                // Add edge from neighbor to vertex (bidirectional)
                if neighbor_id < graph.len() && !graph[neighbor_id].contains(&vertex_id) {
                    graph[neighbor_id].push(vertex_id);
                }
            }
        }
        
        drop(graph);
        
        // Prune neighbors of affected vertices
        for &neighbor_id in &candidates {
            self.prune_vertex_dynamic(neighbor_id, vectors)?;
        }
        
        // Prune the new vertex
        self.prune_vertex_dynamic(vertex_id, vectors)?;
        
        Ok(())
    }
    
    /// Delete a vertex by removing all edges to/from it
    pub fn delete_vertex(&self, vertex_id: usize) -> Result<()> {
        let mut graph = self.graph.write();
        
        if vertex_id >= graph.len() {
            return Ok(()); // Already deleted or never existed
        }
        
        // Get neighbors before clearing
        let neighbors = graph[vertex_id].clone();
        
        // Clear this vertex's edges
        graph[vertex_id].clear();
        
        // Remove edges from neighbors to this vertex
        for &neighbor_id in &neighbors {
            if neighbor_id < graph.len() {
                graph[neighbor_id].retain(|&id| id != vertex_id);
            }
        }
        
        // Update entry point if necessary
        if vertex_id == self.entry_point.load(AtomicOrdering::Relaxed) {
            // Find a new entry point among neighbors
            for &neighbor_id in &neighbors {
                if neighbor_id < graph.len() && !graph[neighbor_id].is_empty() {
                    // This is a simple heuristic; in production, you might want to recalculate the medoid
                    drop(graph);
                    self.entry_point.store(neighbor_id, AtomicOrdering::Relaxed);
                    return Ok(());
                }
            }
        }
        
        Ok(())
    }
    
    /// Set a new entry point
    pub fn set_entry_point(&self, new_entry: usize) {
        self.entry_point.store(new_entry, AtomicOrdering::Relaxed);
    }
    
    /// Greedy search for dynamic index (handles Option<Vec<f32>>)
    fn greedy_search_dynamic(&self, query: &[f32], k: usize, vectors: &[Option<Vec<f32>>]) -> Result<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        
        // Find a valid entry point
        let mut entry = self.entry_point.load(AtomicOrdering::Relaxed);
        if entry >= vectors.len() || vectors[entry].is_none() {
            // Find first valid vector
            for (i, v) in vectors.iter().enumerate() {
                if v.is_some() {
                    entry = i;
                    break;
                }
            }
        }
        
        if entry >= vectors.len() || vectors[entry].is_none() {
            return Ok(vec![]); // No valid vectors
        }
        
        // Start from entry point
        let entry_dist = self.distance_fn.distance(query, vectors[entry].as_ref().unwrap())?;
        candidates.push(Neighbor { id: entry, distance: entry_dist });
        w.push(Neighbor { id: entry, distance: entry_dist });
        visited.insert(entry);
        
        while !candidates.is_empty() {
            let current = candidates.pop().unwrap();
            
            // Check termination condition
            if current.distance > w.peek().unwrap().distance * self.alpha {
                break;
            }
            
            // Explore neighbors
            let graph = self.graph.read();
            let neighbors = if current.id < graph.len() {
                graph[current.id].clone()
            } else {
                vec![]
            };
            drop(graph);
            
            for &neighbor_id in &neighbors {
                if !visited.contains(&neighbor_id) && neighbor_id < vectors.len() {
                    if let Some(neighbor_vec) = &vectors[neighbor_id] {
                        visited.insert(neighbor_id);
                        
                        let dist = self.distance_fn.distance(query, neighbor_vec)?;
                        let neighbor = Neighbor { id: neighbor_id, distance: dist };
                        
                        if dist < w.peek().unwrap().distance || w.len() < k {
                            candidates.push(neighbor);
                            w.push(neighbor);
                            
                            // Maintain size k
                            if w.len() > k {
                                w.pop();
                            }
                        }
                    }
                }
            }
        }
        
        // Extract result
        let mut result: Vec<usize> = w.into_iter().map(|n| n.id).collect();
        result.truncate(k);
        Ok(result)
    }
    
    /// Prune vertex for dynamic index
    fn prune_vertex_dynamic(&self, vertex_id: usize, vectors: &[Option<Vec<f32>>]) -> Result<()> {
        if vertex_id >= vectors.len() || vectors[vertex_id].is_none() {
            return Ok(());
        }
        
        let graph = self.graph.read();
        if vertex_id >= graph.len() {
            return Ok(());
        }
        
        let candidates: Vec<usize> = graph[vertex_id].clone();
        drop(graph);
        
        if candidates.len() <= self.max_degree {
            return Ok(());
        }
        
        let vertex_vec = vectors[vertex_id].as_ref().unwrap();
        
        // Calculate distances to all candidates
        let mut neighbors: Vec<Neighbor> = Vec::new();
        for &id in &candidates {
            if id < vectors.len() {
                if let Some(neighbor_vec) = &vectors[id] {
                    let dist = self.distance_fn.distance(vertex_vec, neighbor_vec)?;
                    neighbors.push(Neighbor { id, distance: dist });
                }
            }
        }
        
        // Sort by distance
        neighbors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        // RobustPrune algorithm
        let mut pruned = Vec::new();
        let mut pruned_set = HashSet::new();
        
        for neighbor in neighbors {
            if pruned.len() >= self.max_degree {
                break;
            }
            
            // Check if this neighbor is closer to vertex than to any selected neighbor
            let mut should_prune = false;
            
            for &selected_id in &pruned {
                if let Some(selected_vec) = vectors.get(selected_id).and_then(|v: &Option<Vec<f32>>| v.as_ref()) {
                    if let Some(neighbor_vec) = vectors.get(neighbor.id).and_then(|v| v.as_ref()) {
                        let dist_to_selected = self.distance_fn.distance(neighbor_vec, selected_vec)?;
                        if dist_to_selected < neighbor.distance {
                            should_prune = true;
                            break;
                        }
                    }
                }
            }
            
            if !should_prune {
                pruned.push(neighbor.id);
                pruned_set.insert(neighbor.id);
            }
        }
        
        // Update graph
        let mut graph = self.graph.write();
        if vertex_id < graph.len() {
            graph[vertex_id] = pruned;
            
            // Remove reverse edges for pruned neighbors
            for &id in &candidates {
                if !pruned_set.contains(&id) && id < graph.len() {
                    graph[id].retain(|&x| x != vertex_id);
                }
            }
        }
        
        Ok(())
    }
}

// Custom serialization implementation for VamanaGraph
impl Serialize for VamanaGraph {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let graph = self.graph.read();
        let serializable = SerializableVamanaGraph {
            graph: graph.clone(),
            max_degree: self.max_degree,
            search_list_size: self.search_list_size,
            alpha: self.alpha,
            num_vertices: self.num_vertices,
            entry_point: self.entry_point.load(AtomicOrdering::Relaxed),
            metric: self.metric,
            dimension: self.dimension,
        };
        serializable.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for VamanaGraph {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serializable = SerializableVamanaGraph::deserialize(deserializer)?;
        Ok(VamanaGraph::from_serializable(serializable))
    }
}

impl VamanaGraph {
    /// Create VamanaGraph from serializable representation
    pub fn from_serializable(serializable: SerializableVamanaGraph) -> Self {
        VamanaGraph {
            graph: Arc::new(RwLock::new(serializable.graph)),
            max_degree: serializable.max_degree,
            search_list_size: serializable.search_list_size,
            alpha: serializable.alpha,
            num_vertices: serializable.num_vertices,
            entry_point: AtomicUsize::new(serializable.entry_point),
            distance_fn: create_distance_function(serializable.metric, serializable.dimension),
            metric: serializable.metric,
            dimension: serializable.dimension,
        }
    }
}

/// Graph statistics
#[derive(Debug)]
pub struct GraphStats {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub avg_degree: f32,
    pub max_degree: usize,
    pub min_degree: usize,
    pub entry_point: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_vamana_build() {
        let vectors = generate_random_vectors(100, 16);
        let mut graph = VamanaGraph::new(100, 16, Distance::L2, 32, 50, 1.2);
        
        graph.build(&vectors).unwrap();
        
        let stats = graph.stats();
        assert_eq!(stats.num_vertices, 100);
        assert!(stats.avg_degree > 0.0);
        assert!(stats.avg_degree <= 32.0);
    }
    
    #[test]
    fn test_vamana_search() {
        let vectors = generate_random_vectors(100, 16);
        let mut graph = VamanaGraph::new(100, 16, Distance::L2, 32, 50, 1.2);
        
        graph.build(&vectors).unwrap();
        
        // Search with a vector from the dataset
        let results = graph.search(&vectors[0], 5, &vectors).unwrap();
        
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, 0); // Should find itself
        assert_eq!(results[0].1, 0.0); // Distance to itself is 0
    }
}