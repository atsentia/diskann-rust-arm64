//! Optimized Vamana implementation with C++ performance parity
use crate::{Distance, DistanceFunction, Result, Error};
use crate::distance::create_distance_function;
use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::cmp::Ordering;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Parameters matching C++ DiskANN defaults
pub struct VamanaParams {
    pub max_degree: usize,           // R (default: 64)
    pub search_list_size: usize,     // L for search (default: 100)  
    pub build_list_size: usize,      // L for build (default: 750 like C++)
    pub alpha: f32,                  // α (default: 1.2)
    pub graph_slack_factor: f32,     // GRAPH_SLACK_FACTOR (default: 1.05)
}

impl Default for VamanaParams {
    fn default() -> Self {
        Self {
            max_degree: 64,
            search_list_size: 100,
            build_list_size: 750,    // C++ DEFAULT_MAXC
            alpha: 1.2,
            graph_slack_factor: 1.05, // C++ defaults::GRAPH_SLACK_FACTOR
        }
    }
}

/// Scratch space for searches to avoid allocations
struct SearchScratch {
    visited: Vec<bool>,
    candidates: Vec<Neighbor>,
    w: Vec<Neighbor>,
    pool: Vec<Neighbor>,
    distances: Vec<f32>,
}

impl SearchScratch {
    fn new(num_vertices: usize, max_candidates: usize) -> Self {
        Self {
            visited: vec![false; num_vertices],
            candidates: Vec::with_capacity(max_candidates * 2),
            w: Vec::with_capacity(max_candidates),
            pool: Vec::with_capacity(max_candidates),
            distances: Vec::with_capacity(max_candidates),
        }
    }
    
    fn reset(&mut self) {
        // Fast reset of visited array
        self.visited.fill(false);
        self.candidates.clear();
        self.w.clear();
        self.pool.clear();
        self.distances.clear();
    }
}

/// Optimized Vamana graph implementation
pub struct VamanaGraphOptimized {
    num_vertices: usize,
    dimension: usize,
    params: VamanaParams,
    distance_fn: Arc<dyn DistanceFunction>,
    graph: Arc<RwLock<Vec<Vec<usize>>>>,
    entry_point: AtomicUsize,
    // Pre-allocated scratch spaces for each thread
    scratch_spaces: parking_lot::RwLock<Vec<SearchScratch>>,
}

impl VamanaGraphOptimized {
    pub fn new(
        num_vertices: usize,
        dimension: usize,
        metric: Distance,
        params: VamanaParams,
    ) -> Self {
        let distance_fn: Arc<dyn DistanceFunction> = Arc::from(create_distance_function(metric, dimension));
        let graph = Arc::new(RwLock::new(vec![Vec::with_capacity(params.max_degree); num_vertices]));
        
        // Pre-allocate scratch spaces for threads
        let num_threads = rayon::current_num_threads();
        let scratch_spaces = (0..num_threads)
            .map(|_| SearchScratch::new(num_vertices, params.build_list_size))
            .collect();
        
        Self {
            num_vertices,
            dimension,
            params,
            distance_fn,
            graph,
            entry_point: AtomicUsize::new(0),
            scratch_spaces: parking_lot::RwLock::new(scratch_spaces),
        }
    }
    
    /// Build graph with optimizations
    pub fn build(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::InvalidParameter("No vectors provided".to_string()).into());
        }
        
        // Calculate entry point (medoid) - using our optimized NEON version
        let start_point = self.calculate_entry_point(vectors)?;
        self.entry_point.store(start_point, AtomicOrdering::Relaxed);
        
        // Create visit order
        let visit_order: Vec<usize> = (0..self.num_vertices).collect();
        
        // Link phase - parallel construction
        self.link_parallel(&visit_order, vectors)?;
        
        // Final cleanup phase
        self.final_cleanup(&visit_order, vectors)?;
        
        Ok(())
    }
    
    /// Optimized parallel link phase
    fn link_parallel(&self, visit_order: &[usize], vectors: &[Vec<f32>]) -> Result<()> {
        use std::time::Instant;
        let start_time = Instant::now();
        
        println!("🔗 Building graph (optimized algorithm)...");
        
        #[cfg(feature = "parallel")]
        {
            let processed = AtomicUsize::new(0);
            let total = visit_order.len();
            
            // Use larger chunks for better load balancing
            const CHUNK_SIZE: usize = 256;
            
            // Get thread ID for scratch space
            let thread_local_scratch = std::cell::RefCell::new(None);
            
            visit_order
                .par_chunks(CHUNK_SIZE)
                .try_for_each(|chunk| -> Result<()> {
                    // Get thread-local scratch space
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    let mut scratch_guard = self.scratch_spaces.write();
                    let scratch = &mut scratch_guard[thread_id];
                    drop(scratch_guard);
                    
                    for &node in chunk {
                        // Search for neighbors using scratch space
                        scratch.reset();
                        let pruned_list = self.search_for_point_and_prune_optimized(
                            node, 
                            self.params.build_list_size, 
                            vectors,
                            scratch
                        )?;
                        
                        // Set neighbors for this node
                        {
                            let mut graph = self.graph.write();
                            graph[node] = pruned_list.clone();
                        }
                        
                        // Inter-insert: add reverse edges
                        self.inter_insert_batch(node, &pruned_list, vectors)?;
                        
                        // Progress reporting
                        let count = processed.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                        if count % 10000 == 0 {
                            println!("   📊 {:.1}% completed", (count as f64 / total as f64) * 100.0);
                        }
                    }
                    Ok(())
                })?;
        }
        
        let elapsed = start_time.elapsed();
        println!("   ✅ Link phase completed in {:.2}s ({:.0} vectors/sec)", 
                 elapsed.as_secs_f64(), 
                 self.num_vertices as f64 / elapsed.as_secs_f64());
        
        Ok(())
    }
    
    /// Optimized search using pre-allocated scratch space
    fn search_for_point_and_prune_optimized(
        &self, 
        location: usize, 
        l_build: usize, 
        vectors: &[Vec<f32>],
        scratch: &mut SearchScratch
    ) -> Result<Vec<usize>> {
        let query = &vectors[location];
        
        // Initialize with entry point
        let entry = self.entry_point.load(AtomicOrdering::Relaxed);
        let entry_dist = self.distance_fn.distance(query, &vectors[entry])?;
        
        scratch.candidates.push(Neighbor { id: entry, distance: entry_dist });
        scratch.w.push(Neighbor { id: entry, distance: entry_dist });
        scratch.visited[entry] = true;
        
        // Main search loop
        while let Some(current) = scratch.candidates.pop() {
            if current.distance > scratch.w[0].distance * self.params.alpha {
                break;
            }
            
            // Get neighbors and compute distances in batch
            let neighbors = {
                let graph = self.graph.read();
                graph[current.id].clone()
            };
            
            // Batch distance computation for unvisited neighbors
            let mut batch_neighbors = Vec::new();
            for &neighbor_id in &neighbors {
                if !scratch.visited[neighbor_id] {
                    batch_neighbors.push(neighbor_id);
                    scratch.visited[neighbor_id] = true;
                }
            }
            
            // Compute distances for batch
            if !batch_neighbors.is_empty() {
                scratch.distances.clear();
                for &neighbor_id in &batch_neighbors {
                    let dist = self.distance_fn.distance(query, &vectors[neighbor_id])?;
                    scratch.distances.push(dist);
                }
                
                // Add to candidates and W
                for (i, &neighbor_id) in batch_neighbors.iter().enumerate() {
                    let neighbor = Neighbor { 
                        id: neighbor_id, 
                        distance: scratch.distances[i] 
                    };
                    
                    scratch.candidates.push(neighbor);
                    
                    // Insert into W maintaining top L
                    scratch.w.push(neighbor);
                    if scratch.w.len() > l_build {
                        // Maintain heap property - remove furthest
                        scratch.w.sort_unstable_by(|a, b| 
                            a.distance.partial_cmp(&b.distance).unwrap()
                        );
                        scratch.w.truncate(l_build);
                    }
                }
            }
        }
        
        // Convert W to pool for pruning
        scratch.pool.clear();
        scratch.pool.extend(scratch.w.iter()
            .filter(|n| n.id != location)
            .copied());
        
        scratch.pool.sort_unstable_by(|a, b| 
            a.distance.partial_cmp(&b.distance).unwrap()
        );
        
        // Prune using RobustPrune algorithm
        let pruned = self.robust_prune_optimized(location, &scratch.pool, vectors)?;
        
        Ok(pruned)
    }
    
    /// Optimized RobustPrune with better cache locality
    fn robust_prune_optimized(
        &self, 
        _location: usize, 
        pool: &[Neighbor], 
        vectors: &[Vec<f32>]
    ) -> Result<Vec<usize>> {
        if pool.is_empty() {
            return Ok(vec![]);
        }
        
        let mut result = Vec::with_capacity(self.params.max_degree);
        result.push(pool[0].id);
        
        // Occlude list algorithm with optimizations
        let mut cur_alpha = 1.0;
        
        while cur_alpha <= self.params.alpha && result.len() < self.params.max_degree {
            let mut next_best = None;
            let mut next_best_dist = f32::MAX;
            
            // Check each candidate
            'candidate_loop: for candidate in pool.iter() {
                if result.contains(&candidate.id) {
                    continue;
                }
                
                // Check occlusion with existing neighbors
                for &existing_id in &result {
                    let exist_to_cand_dist = self.distance_fn.distance(
                        &vectors[existing_id], 
                        &vectors[candidate.id]
                    )?;
                    
                    if cur_alpha * exist_to_cand_dist < candidate.distance {
                        continue 'candidate_loop;
                    }
                }
                
                // Not occluded - check if best
                if candidate.distance < next_best_dist {
                    next_best = Some(candidate);
                    next_best_dist = candidate.distance;
                }
            }
            
            if let Some(best) = next_best {
                result.push(best.id);
            } else {
                cur_alpha *= 1.2;
            }
        }
        
        Ok(result)
    }
    
    /// Batch inter-insert for better performance
    fn inter_insert_batch(
        &self, 
        node: usize, 
        pruned_list: &[usize], 
        vectors: &[Vec<f32>]
    ) -> Result<()> {
        // Process inter-insertions in batch
        let mut updates = Vec::new();
        
        for &neighbor in pruned_list {
            let mut prune_needed = false;
            let mut neighbors_to_prune = Vec::new();
            
            {
                let mut graph = self.graph.write();
                let neighbor_list = &mut graph[neighbor];
                
                if !neighbor_list.contains(&node) {
                    if neighbor_list.len() < (self.params.graph_slack_factor * self.params.max_degree as f32) as usize {
                        neighbor_list.push(node);
                    } else {
                        neighbors_to_prune = neighbor_list.clone();
                        neighbors_to_prune.push(node);
                        prune_needed = true;
                    }
                }
            }
            
            if prune_needed {
                updates.push((neighbor, neighbors_to_prune));
            }
        }
        
        // Process all pruning operations
        for (neighbor, neighbors_to_prune) in updates {
            let mut pool = Vec::with_capacity(neighbors_to_prune.len());
            
            for &nbr in &neighbors_to_prune {
                if nbr != neighbor {
                    let dist = self.distance_fn.distance(&vectors[neighbor], &vectors[nbr])?;
                    pool.push(Neighbor { id: nbr, distance: dist });
                }
            }
            
            pool.sort_unstable_by(|a, b| 
                a.distance.partial_cmp(&b.distance).unwrap()
            );
            
            let pruned = self.robust_prune_optimized(neighbor, &pool, vectors)?;
            
            let mut graph = self.graph.write();
            graph[neighbor] = pruned;
        }
        
        Ok(())
    }
    
    /// Final cleanup phase
    fn final_cleanup(&self, visit_order: &[usize], vectors: &[Vec<f32>]) -> Result<()> {
        println!("   🧹 Starting final cleanup...");
        
        #[cfg(feature = "parallel")]
        {
            visit_order
                .par_chunks(256)
                .try_for_each(|chunk| -> Result<()> {
                    for &node in chunk {
                        let needs_pruning = {
                            let graph = self.graph.read();
                            graph[node].len() > self.params.max_degree
                        };
                        
                        if needs_pruning {
                            self.cleanup_node(node, vectors)?;
                        }
                    }
                    Ok(())
                })?;
        }
        
        Ok(())
    }
    
    /// Cleanup a single node
    fn cleanup_node(&self, node: usize, vectors: &[Vec<f32>]) -> Result<()> {
        let neighbors = {
            let graph = self.graph.read();
            graph[node].clone()
        };
        
        let mut pool = Vec::with_capacity(neighbors.len());
        for &neighbor in &neighbors {
            let dist = self.distance_fn.distance(&vectors[node], &vectors[neighbor])?;
            pool.push(Neighbor { id: neighbor, distance: dist });
        }
        
        pool.sort_unstable_by(|a, b| 
            a.distance.partial_cmp(&b.distance).unwrap()
        );
        
        let pruned = self.robust_prune_optimized(node, &pool, vectors)?;
        
        let mut graph = self.graph.write();
        graph[node] = pruned;
        
        Ok(())
    }
    
    /// Calculate entry point using NEON-optimized medoid
    fn calculate_entry_point(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        self.find_medoid_centroid_neon(vectors)
    }
    
    /// NEON-optimized medoid calculation
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn find_medoid_centroid_neon(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        use std::arch::aarch64::*;
        
        let num_vectors = vectors.len();
        let dimension = self.dimension;
        
        // Calculate centroid using NEON
        let mut centroid = vec![0.0f32; dimension];
        
        unsafe {
            for vector in vectors {
                let mut i = 0;
                
                // Process 4 elements at a time
                while i + 4 <= dimension {
                    let vec_chunk = vld1q_f32(vector[i..].as_ptr());
                    let cent_chunk = vld1q_f32(centroid[i..].as_ptr());
                    let sum = vaddq_f32(cent_chunk, vec_chunk);
                    vst1q_f32(centroid[i..].as_mut_ptr(), sum);
                    i += 4;
                }
                
                // Handle remaining elements
                while i < dimension {
                    centroid[i] += vector[i];
                    i += 1;
                }
            }
            
            // Divide by number of vectors
            let inv_n = vdupq_n_f32(1.0 / num_vectors as f32);
            let mut i = 0;
            
            while i + 4 <= dimension {
                let cent_chunk = vld1q_f32(centroid[i..].as_ptr());
                let avg = vmulq_f32(cent_chunk, inv_n);
                vst1q_f32(centroid[i..].as_mut_ptr(), avg);
                i += 4;
            }
            
            while i < dimension {
                centroid[i] /= num_vectors as f32;
                i += 1;
            }
        }
        
        // Find closest point to centroid
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
    
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    fn find_medoid_centroid_neon(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        // Fallback to scalar version
        self.find_medoid_centroid(vectors)
    }
    
    /// Scalar medoid calculation
    fn find_medoid_centroid(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        let num_vectors = vectors.len();
        let dimension = self.dimension;
        
        // Calculate centroid
        let mut centroid = vec![0.0f32; dimension];
        for vector in vectors {
            for (i, &val) in vector.iter().enumerate() {
                centroid[i] += val;
            }
        }
        
        let inv_n = 1.0 / num_vectors as f32;
        for val in centroid.iter_mut() {
            *val *= inv_n;
        }
        
        // Find closest point to centroid
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
            min_degree: if min_degree == usize::MAX { 0 } else { min_degree },
            entry_point: self.entry_point.load(AtomicOrdering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Neighbor {
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
        other.distance.partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub avg_degree: f32,
    pub max_degree: usize,
    pub min_degree: usize,
    pub entry_point: usize,
}