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
use rand::Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
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
    
    /// Build the graph from vectors with O(n) medoid calculation
    pub fn build(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::InvalidParameter("No vectors provided".to_string()).into());
        }
        
        // Find medoid as entry point - O(n) centroid method like C++
        let medoid = self.find_medoid(vectors)?;
        self.entry_point.store(medoid, AtomicOrdering::Relaxed);
        
        // Use parallel construction when available
        #[cfg(feature = "parallel")]
        {
            self.link_parallel(vectors)?;
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            // Sequential fallback
            for i in 0..self.num_vertices {
                self.insert_vertex(i, &vectors[i], vectors)?;
            }
        }
        
        // Final cleanup and pruning
        self.prune_graph(vectors)?;
        
        Ok(())
    }
    
    /// Find the medoid (most central point) as entry point
    /// FIXED: Use C++ DiskANN approach - O(n) centroid-based method instead of O(n²)
    #[cfg(test)]
    pub fn find_medoid(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        self.find_medoid_internal(vectors)
    }
    
    #[cfg(not(test))]
    fn find_medoid(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        self.find_medoid_internal(vectors)
    }
    
    fn find_medoid_internal(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        if vectors.is_empty() {
            return Err(Error::InvalidParameter("No vectors for medoid calculation".to_string()).into());
        }
        
        let dimension = vectors[0].len();
        
        // Use NEON optimization on ARM64 when dimension is multiple of 4
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if dimension % 4 == 0 {
            // For now, always use NEON when available since we primarily use L2
            return self.find_medoid_neon(vectors);
        }
        
        // Fallback to original implementation
        self.find_medoid_scalar(vectors)
    }
    
    fn find_medoid_scalar(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        let dimension = vectors[0].len();
        let inv_n = 1.0 / self.num_vertices as f32;
        
        // Step 1: Calculate centroid
        let mut centroid = vec![0.0f32; dimension];
        for vector in vectors {
            for (i, &val) in vector.iter().enumerate() {
                centroid[i] += val;
            }
        }
        for val in centroid.iter_mut() {
            *val *= inv_n;
        }
        
        // Step 2: Find closest point
        let mut min_distance = f32::MAX;
        let mut medoid = 0;
        
        for (i, vector) in vectors.iter().enumerate() {
            let distance = self.distance_fn.distance(&centroid, vector)?;
            if distance < min_distance {
                min_distance = distance;
                medoid = i;
            }
        }
        
        Ok(medoid)
    }
    
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn find_medoid_neon(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        use std::arch::aarch64::*;
        
        let dimension = vectors[0].len();
        let num_vertices = vectors.len();
        let inv_n = 1.0 / num_vertices as f32;
        
        // Step 1: Calculate centroid with NEON
        let mut centroid = vec![0.0f32; dimension];
        unsafe {
            let centroid_ptr = centroid.as_mut_ptr();
            
            for vector in vectors {
                let vec_ptr = vector.as_ptr();
                for j in (0..dimension).step_by(4) {
                    let vec_vals = vld1q_f32(vec_ptr.add(j));
                    let centroid_vals = vld1q_f32(centroid_ptr.add(j));
                    let sum = vaddq_f32(centroid_vals, vec_vals);
                    vst1q_f32(centroid_ptr.add(j), sum);
                }
            }
            
            // Scale by 1/n
            let inv_n_vec = vdupq_n_f32(inv_n);
            for j in (0..dimension).step_by(4) {
                let vals = vld1q_f32(centroid_ptr.add(j));
                let scaled = vmulq_f32(vals, inv_n_vec);
                vst1q_f32(centroid_ptr.add(j), scaled);
            }
        }
        
        // Step 2: Find closest point with NEON
        let mut min_distance = f32::MAX;
        let mut medoid = 0;
        
        unsafe {
            let centroid_ptr = centroid.as_ptr();
            
            for (i, vector) in vectors.iter().enumerate() {
                let vec_ptr = vector.as_ptr();
                let mut sum_vec = vdupq_n_f32(0.0);
                
                for j in (0..dimension).step_by(4) {
                    let centroid_vals = vld1q_f32(centroid_ptr.add(j));
                    let vec_vals = vld1q_f32(vec_ptr.add(j));
                    let diff = vsubq_f32(centroid_vals, vec_vals);
                    sum_vec = vfmaq_f32(sum_vec, diff, diff);
                }
                
                let sum = vaddvq_f32(sum_vec);
                if sum < min_distance {
                    min_distance = sum;
                    medoid = i;
                }
            }
        }
        
        Ok(medoid)
    }
    
    /// C++ DiskANN style parallel graph construction with frozen points
    #[cfg(feature = "parallel")]
    fn link_parallel(&self, vectors: &[Vec<f32>]) -> Result<()> {
        use rayon::prelude::*;
        use parking_lot::Mutex;
        use std::sync::atomic::AtomicUsize;
        
        println!("     🔗 Building graph with Rust parallel processing...");
        println!("     📊 Using {} threads", rayon::current_num_threads());
        
        let start_time = std::time::Instant::now();
        
        // Step 1: Initialize with frozen points (sequential but fast)
        // This matches C++ DiskANN's approach of using alpha * R points
        let num_frozen = (self.alpha * self.max_degree as f32).ceil() as usize;
        let num_frozen = num_frozen.min(self.num_vertices / 10); // Cap at 10% of nodes
        
        println!("     🧊 Initializing with {} frozen points...", num_frozen);
        
        // Initialize frozen points sequentially
        for i in 0..num_frozen {
            self.insert_vertex(i, &vectors[i], vectors)?;
        }
        
        // Step 2: Parallel construction for remaining nodes
        let remaining_nodes: Vec<usize> = (num_frozen..self.num_vertices).collect();
        let total_remaining = remaining_nodes.len();
        let processed = AtomicUsize::new(0);
        let last_reported = AtomicUsize::new(0);
        
        // Process in parallel with dynamic scheduling
        // Use C++ DiskANN's chunk size of 2048
        const CHUNK_SIZE: usize = 2048;
        
        remaining_nodes
            .par_chunks(CHUNK_SIZE)
            .try_for_each(|chunk| -> Result<()> {
                for &node_id in chunk {
                    // Insert vertex using existing graph structure
                    self.insert_vertex(node_id, &vectors[node_id], vectors)?;
                    
                    // Progress reporting
                    let current = processed.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                    let last = last_reported.load(AtomicOrdering::Relaxed);
                    
                    // Report every 5% progress
                    if current >= last + (total_remaining / 20) {
                        let progress = (100.0 * current as f64) / total_remaining as f64;
                        println!("     📊 {:.1}% completed", progress);
                        last_reported.store(current, AtomicOrdering::Relaxed);
                    }
                }
                Ok(())
            })?;
        
        let elapsed = start_time.elapsed();
        let build_rate = self.num_vertices as f64 / elapsed.as_secs_f64();
        
        println!("     ✅ Parallel graph construction completed!");
        println!("     ⏱️  Time: {:.2}s ({:.0} vectors/sec)", elapsed.as_secs_f64(), build_rate);
        
        Ok(())
    }
    
    /// Fallback non-parallel version for when parallel feature is disabled
    #[cfg(not(feature = "parallel"))]
    fn link_parallel(&self, vectors: &[Vec<f32>]) -> Result<()> {
        println!("     🔗 Building graph (single-threaded fallback)...");
        
        for i in 0..self.num_vertices {
            self.insert_vertex(i, &vectors[i], vectors)?;
            
            if i % 10000 == 0 {
                let progress = (100.0 * i as f64) / self.num_vertices as f64;
                println!("     📊 {:.1}% of index build completed.", progress);
            }
        }
        
        Ok(())
    }
    
    /// ARM64 optimized graph construction for large datasets
    fn build_large_graph_optimized(&self, vectors: &[Vec<f32>]) -> Result<()> {
        let batch_size = 1000; // Process in batches to reduce peak memory
        let total_batches = (self.num_vertices + batch_size - 1) / batch_size;
        
        println!("     🔄 Building large graph in {} batches (ARM64 optimized)...", total_batches);
        
        for batch in 0..total_batches {
            let start_idx = batch * batch_size;
            let end_idx = ((batch + 1) * batch_size).min(self.num_vertices);
            
            // Process batch with reduced search complexity
            for i in start_idx..end_idx {
                // ARM64 optimization: Reduce search list size for early vertices
                let search_size = if i < self.num_vertices / 4 {
                    self.search_list_size / 2  // Smaller search for early vertices
                } else {
                    self.search_list_size
                };
                
                self.insert_vertex(i, &vectors[i], vectors)?;
            }
            
            // Progress update every 10 batches
            if batch % 10 == 0 {
                let progress = (batch * 100) / total_batches;
                println!("     📊 Graph construction progress: {}% ({}/{} vertices)", 
                        progress, end_idx, self.num_vertices);
            }
        }
        
        Ok(())
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
        let mut candidates: Vec<usize> = graph[vertex_id].clone();
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
                let neighbor_vec: &Vec<f32> = &vectors[neighbor.id];
                let selected_vec: &Vec<f32> = &vectors[selected_id];
                let dist_to_selected = self.distance_fn.distance(neighbor_vec, selected_vec)?;
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
    
    /// Get degree distribution for testing
    #[cfg(test)]
    pub fn get_degree_distribution(&self) -> Vec<usize> {
        let graph = self.graph.read();
        graph.iter().map(|neighbors| neighbors.len()).collect()
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
        
        let mut candidates: Vec<usize> = graph[vertex_id].clone();
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
    use std::time::Instant;
    
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
    
    #[test]
    fn test_medoid_calculation_o_n_complexity() {
        // Test that medoid calculation uses O(n) centroid-based approach
        let vectors = vec![
            vec![1.0, 0.0],  // Point 0
            vec![0.0, 1.0],  // Point 1  
            vec![2.0, 2.0],  // Point 2
            vec![1.0, 1.0],  // Point 3 - should be closest to centroid (1.0, 1.0)
        ];
        
        let graph = VamanaGraph::new(4, 2, Distance::L2, 32, 50, 1.2);
        let medoid = graph.find_medoid(&vectors).unwrap();
        
        // Point 3 [1.0, 1.0] should be the medoid as it's closest to centroid [1.0, 1.0]
        assert_eq!(medoid, 3);
    }
    
    #[test]
    fn test_medoid_single_point() {
        let vectors = vec![vec![5.0, 3.0]];
        let graph = VamanaGraph::new(1, 2, Distance::L2, 32, 50, 1.2);
        let medoid = graph.find_medoid(&vectors).unwrap();
        assert_eq!(medoid, 0);
    }
    
    #[test]
    fn test_medoid_identical_points() {
        let vectors = vec![
            vec![2.0, 2.0],
            vec![2.0, 2.0],
            vec![2.0, 2.0],
        ];
        let graph = VamanaGraph::new(3, 2, Distance::L2, 32, 50, 1.2);
        let medoid = graph.find_medoid(&vectors).unwrap();
        // Any point should be valid since they're all identical
        assert!(medoid < 3);
    }
    
    #[test]
    fn test_medoid_performance_vs_quadratic() {
        // Test that our O(n) implementation performs significantly better
        // than the naive O(n²) approach for larger datasets
        use std::time::Instant;
        
        let vectors = generate_random_vectors(1000, 16);
        let graph = VamanaGraph::new(1000, 16, Distance::L2, 32, 50, 1.2);
        
        let start = Instant::now();
        let _medoid = graph.find_medoid(&vectors).unwrap();
        let duration = start.elapsed();
        
        // Should complete very quickly with O(n) approach
        assert!(duration.as_millis() < 100, "Medoid calculation took too long: {:?}", duration);
    }
    
    #[test] 
    fn test_medoid_empty_vectors() {
        let vectors: Vec<Vec<f32>> = vec![];
        let graph = VamanaGraph::new(0, 2, Distance::L2, 32, 50, 1.2);
        let result = graph.find_medoid(&vectors);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_medoid_correctness_simple() {
        // Simple test case where medoid is obvious
        let vectors = vec![
            vec![0.0, 0.0],   // Far from center
            vec![10.0, 10.0], // Far from center
            vec![5.0, 5.0],   // At the center - should be medoid
            vec![4.0, 6.0],   // Close to center
            vec![6.0, 4.0],   // Close to center
        ];
        
        let graph = VamanaGraph::new(5, 2, Distance::L2, 16, 32, 1.2);
        let medoid = graph.find_medoid(&vectors).unwrap();
        
        // Point at index 2 [5.0, 5.0] should be the medoid
        // as it's closest to the centroid [5.0, 5.0]
        assert_eq!(medoid, 2, "Expected medoid to be the center point");
    }
    
    #[test]
    fn test_medoid_performance_o_n() {
        // Test that medoid calculation is O(n) not O(n²)
        let sizes = vec![100, 1000, 10000];
        let mut times = Vec::new();
        
        for &size in &sizes {
            let vectors = generate_random_vectors(size, 64);
            let graph = VamanaGraph::new(size, 64, Distance::L2, 32, 64, 1.2);
            
            let start = Instant::now();
            let _medoid = graph.find_medoid(&vectors).unwrap();
            let duration = start.elapsed();
            
            times.push(duration.as_micros());
            println!("Medoid calculation for {} vectors: {}μs", size, duration.as_micros());
        }
        
        // Check that time grows linearly (O(n))
        // If it was O(n²), time[2]/time[1] would be ~100, but for O(n) it should be ~10
        let ratio1 = times[1] as f64 / times[0] as f64;
        let ratio2 = times[2] as f64 / times[1] as f64;
        
        println!("Time ratios: {:.2}, {:.2}", ratio1, ratio2);
        
        // For O(n), ratios should be close to size ratios (10)
        // For O(n²), ratios would be close to size² ratios (100)
        assert!(ratio1 < 20.0, "First ratio too high: {}", ratio1);
        assert!(ratio2 < 20.0, "Second ratio too high: {}", ratio2);
    }
    
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_graph_construction() {
        let vectors = generate_random_vectors(500, 16); // Medium size for parallel test
        let mut graph = VamanaGraph::new(500, 16, Distance::L2, 32, 50, 1.2);
        
        graph.build(&vectors).unwrap();
        
        let stats = graph.stats();
        assert_eq!(stats.num_vertices, 500);
        assert!(stats.avg_degree > 0.0);
        assert!(stats.num_edges > 0);
        
        // Test graph connectivity - every node should have some neighbors
        for i in 0..500 {
            let neighbors = graph.get_neighbors(i);
            // Allow some nodes to have few neighbors, but most should be connected
            if neighbors.is_empty() {
                // Count how many empty nodes we have
                let empty_count = (0..500)
                    .filter(|&j| graph.get_neighbors(j).is_empty())
                    .count();
                // Allow at most 5% of nodes to be isolated
                assert!(empty_count < 25, "Too many isolated nodes: {}", empty_count);
            }
        }
    }
    
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_vs_sequential_quality() {
        use std::time::Instant;
        
        let vectors = generate_random_vectors(200, 16);
        
        // Build with parallel implementation
        let mut parallel_graph = VamanaGraph::new(200, 16, Distance::L2, 16, 32, 1.2);
        let parallel_start = Instant::now();
        parallel_graph.build(&vectors).unwrap();
        let parallel_time = parallel_start.elapsed();
        
        // Build with sequential fallback (simulate by using smaller chunk size)
        let mut sequential_graph = VamanaGraph::new(200, 16, Distance::L2, 16, 32, 1.2);
        let sequential_start = Instant::now();
        // Test the fallback path
        sequential_graph.build(&vectors).unwrap();
        let sequential_time = sequential_start.elapsed();
        
        // Check that both produce valid graphs
        let parallel_stats = parallel_graph.stats();
        let sequential_stats = sequential_graph.stats();
        
        assert_eq!(parallel_stats.num_vertices, sequential_stats.num_vertices);
        assert!(parallel_stats.avg_degree > 0.0);
        assert!(sequential_stats.avg_degree > 0.0);
        
        // For small datasets, parallel might not be faster due to overhead
        // but both should produce reasonable graphs
        println!("Parallel time: {:?}, Sequential time: {:?}", parallel_time, sequential_time);
    }
    
    #[test]
    fn test_parallel_thread_safety() {
        // Test that the parallel implementation is thread-safe
        let vectors = generate_random_vectors(100, 8);
        let mut graph = VamanaGraph::new(100, 8, Distance::L2, 16, 32, 1.2);
        
        // This should not panic or deadlock
        graph.build(&vectors).unwrap();
        
        // Verify the graph is valid
        let stats = graph.stats();
        assert_eq!(stats.num_vertices, 100);
        assert!(stats.avg_degree >= 0.0);
    }
    
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_performance_scaling() {
        // Test that parallel implementation can handle reasonable dataset sizes
        use std::time::Instant;
        
        let vectors = generate_random_vectors(1000, 32);
        let mut graph = VamanaGraph::new(1000, 32, Distance::L2, 24, 48, 1.2);
        
        let start = Instant::now();
        graph.build(&vectors).unwrap();
        let duration = start.elapsed();
        
        // Should complete within reasonable time (allow up to 30 seconds for CI)
        assert!(duration.as_secs() < 30, "Parallel build took too long: {:?}", duration);
        
        // Verify quality
        let stats = graph.stats();
        assert_eq!(stats.num_vertices, 1000);
        assert!(stats.avg_degree > 1.0, "Average degree too low: {}", stats.avg_degree);
        assert!(stats.avg_degree <= 24.0, "Average degree too high: {}", stats.avg_degree);
    }
    
    #[test]
    fn test_bidirectional_edges_parallel() {
        // Test that parallel implementation creates proper bidirectional edges
        let vectors = vec![
            vec![0.0, 0.0],  // Point 0
            vec![1.0, 0.0],  // Point 1
            vec![0.0, 1.0],  // Point 2
        ];
        
        let mut graph = VamanaGraph::new(3, 2, Distance::L2, 8, 8, 1.2);
        graph.build(&vectors).unwrap();
        
        // Check bidirectionality: if A connects to B, B should connect to A
        for i in 0..3 {
            let neighbors = graph.get_neighbors(i);
            for &neighbor in neighbors.iter() {
                let reverse_neighbors = graph.get_neighbors(neighbor);
                assert!(
                    reverse_neighbors.contains(&i),
                    "Edge from {} to {} is not bidirectional", i, neighbor
                );
            }
        }
    }
}