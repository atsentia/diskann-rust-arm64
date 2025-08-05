//! Fixed Vamana implementation matching C++ DiskANN exactly
use crate::{Distance, DistanceFunction, Result, Error};
use crate::distance::create_distance_function;
use parking_lot::RwLock;
use std::collections::BinaryHeap;
use hashbrown::HashSet;
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

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

/// Fixed Vamana graph implementation
pub struct VamanaGraphFixed {
    num_vertices: usize,
    dimension: usize,
    params: VamanaParams,
    distance_fn: Arc<dyn DistanceFunction>,
    graph: Arc<RwLock<Vec<Vec<usize>>>>,
    entry_point: AtomicUsize,
}

impl VamanaGraphFixed {
    pub fn new(
        num_vertices: usize,
        dimension: usize,
        metric: Distance,
        params: VamanaParams,
    ) -> Self {
        let distance_fn: Arc<dyn DistanceFunction> = Arc::from(create_distance_function(metric, dimension));
        let graph = Arc::new(RwLock::new(vec![Vec::new(); num_vertices]));
        
        Self {
            num_vertices,
            dimension,
            params,
            distance_fn,
            graph,
            entry_point: AtomicUsize::new(0),
        }
    }
    
    /// Build graph matching C++ algorithm exactly
    pub fn build(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::InvalidParameter("No vectors provided".to_string()).into());
        }
        
        // Calculate entry point (medoid)
        let start_point = self.calculate_entry_point(vectors)?;
        self.entry_point.store(start_point, AtomicOrdering::Relaxed);
        
        // Create visit order
        let visit_order: Vec<usize> = (0..self.num_vertices).collect();
        
        // Link phase - parallel construction like C++
        self.link(&visit_order, vectors)?;
        
        // Final cleanup phase
        self.final_cleanup(&visit_order, vectors)?;
        
        Ok(())
    }
    
    /// Link phase matching C++ exactly
    fn link(&self, visit_order: &[usize], vectors: &[Vec<f32>]) -> Result<()> {
        use std::time::Instant;
        let start_time = Instant::now();
        
        println!("🔗 Building graph (C++ compatible algorithm)...");
        
        #[cfg(feature = "parallel")]
        {
            let processed = AtomicUsize::new(0);
            let total = visit_order.len();
            
            // Use C++ chunk size
            const CHUNK_SIZE: usize = 2048;
            
            visit_order
                .par_chunks(CHUNK_SIZE)
                .try_for_each(|chunk| -> Result<()> {
                    for &node in chunk {
                        // Search for neighbors using large L (like C++)
                        let mut pruned_list = self.search_for_point_and_prune(
                            node, 
                            self.params.build_list_size, 
                            vectors
                        )?;
                        
                        // Set neighbors for this node
                        {
                            let mut graph = self.graph.write();
                            graph[node] = pruned_list.clone();
                        }
                        
                        // Inter-insert: add reverse edges
                        self.inter_insert(node, &mut pruned_list, vectors)?;
                        
                        // Progress reporting
                        let count = processed.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                        if count % 10000 == 0 {
                            println!("   📊 {:.1}% completed", (count as f64 / total as f64) * 100.0);
                        }
                    }
                    Ok(())
                })?;
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            for &node in visit_order {
                let mut pruned_list = self.search_for_point_and_prune(
                    node, 
                    self.params.build_list_size, 
                    vectors
                )?;
                
                {
                    let mut graph = self.graph.write();
                    graph[node] = pruned_list.clone();
                }
                
                self.inter_insert(node, &mut pruned_list, vectors)?;
            }
        }
        
        let elapsed = start_time.elapsed();
        println!("   ✅ Link phase completed in {:.2}s ({:.0} vectors/sec)", 
                 elapsed.as_secs_f64(), 
                 self.num_vertices as f64 / elapsed.as_secs_f64());
        
        Ok(())
    }
    
    /// Search for point and prune - matches C++ iterate_to_fixed_point + prune
    fn search_for_point_and_prune(
        &self, 
        location: usize, 
        l_build: usize, 
        vectors: &[Vec<f32>]
    ) -> Result<Vec<usize>> {
        let query = &vectors[location];
        
        // Initialize with entry point
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        
        let entry = self.entry_point.load(AtomicOrdering::Relaxed);
        let entry_dist = self.distance_fn.distance(query, &vectors[entry])?;
        
        candidates.push(Neighbor { id: entry, distance: entry_dist });
        w.push(Neighbor { id: entry, distance: entry_dist });
        visited.insert(entry);
        
        // Main search loop
        while !candidates.is_empty() {
            let current = candidates.pop().unwrap();
            
            if current.distance > w.peek().unwrap().distance * self.params.alpha {
                break;
            }
            
            // Check neighbors
            let neighbors = {
                let graph = self.graph.read();
                graph[current.id].clone()
            };
            
            for &neighbor_id in &neighbors {
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id);
                    let dist = self.distance_fn.distance(query, &vectors[neighbor_id])?;
                    let neighbor = Neighbor { id: neighbor_id, distance: dist };
                    
                    candidates.push(neighbor.clone());
                    
                    // Update W (keep top L)
                    w.push(neighbor);
                    if w.len() > l_build {
                        // Remove furthest point
                        let mut temp: Vec<_> = w.into_iter().collect();
                        temp.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                        temp.truncate(l_build);
                        w = temp.into_iter().collect();
                    }
                }
            }
        }
        
        // Convert W to vector for pruning
        let mut pool: Vec<_> = w.into_iter()
            .filter(|n| n.id != location)  // Remove self
            .collect();
        
        pool.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        // Prune using RobustPrune algorithm
        let pruned = self.robust_prune(location, pool, vectors)?;
        
        Ok(pruned)
    }
    
    /// RobustPrune algorithm (occlude_list in C++)
    fn robust_prune(
        &self, 
        location: usize, 
        mut pool: Vec<Neighbor>, 
        vectors: &[Vec<f32>]
    ) -> Result<Vec<usize>> {
        if pool.is_empty() {
            return Ok(vec![]);
        }
        
        // Start with nearest neighbor
        let mut result = vec![pool[0].id];
        
        // Occlude list algorithm
        let mut cur_alpha = 1.0;
        while cur_alpha <= self.params.alpha && result.len() < self.params.max_degree {
            let mut next_best_idx = None;
            let mut next_best_dist = f32::MAX;
            
            // Find next best non-occluded candidate
            for (idx, candidate) in pool.iter().enumerate() {
                if result.contains(&candidate.id) {
                    continue;
                }
                
                let mut occluded = false;
                for &existing_id in &result {
                    let exist_to_cand_dist = self.distance_fn.distance(
                        &vectors[existing_id], 
                        &vectors[candidate.id]
                    )?;
                    
                    // Check occlusion condition
                    if cur_alpha * exist_to_cand_dist < candidate.distance {
                        occluded = true;
                        break;
                    }
                }
                
                if !occluded && candidate.distance < next_best_dist {
                    next_best_idx = Some(idx);
                    next_best_dist = candidate.distance;
                }
            }
            
            if let Some(idx) = next_best_idx {
                result.push(pool[idx].id);
            } else {
                // Increase alpha and try again
                cur_alpha *= 1.2;
            }
        }
        
        Ok(result)
    }
    
    /// Inter-insert: add bidirectional edges like C++
    fn inter_insert(
        &self, 
        node: usize, 
        pruned_list: &mut Vec<usize>, 
        vectors: &[Vec<f32>]
    ) -> Result<()> {
        for &neighbor in pruned_list.iter() {
            let mut prune_needed = false;
            let mut neighbors_copy = Vec::new();
            
            {
                let mut graph = self.graph.write();
                let neighbor_list = &mut graph[neighbor];
                
                if !neighbor_list.contains(&node) {
                    if neighbor_list.len() < (self.params.graph_slack_factor * self.params.max_degree as f32) as usize {
                        // Add directly
                        neighbor_list.push(node);
                    } else {
                        // Need to prune
                        neighbors_copy = neighbor_list.clone();
                        neighbors_copy.push(node);
                        prune_needed = true;
                    }
                }
            }
            
            // Prune if needed
            if prune_needed {
                let mut pool = Vec::new();
                for &nbr in &neighbors_copy {
                    if nbr != neighbor {
                        let dist = self.distance_fn.distance(&vectors[neighbor], &vectors[nbr])?;
                        pool.push(Neighbor { id: nbr, distance: dist });
                    }
                }
                
                pool.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                let pruned = self.robust_prune(neighbor, pool, vectors)?;
                
                let mut graph = self.graph.write();
                graph[neighbor] = pruned;
            }
        }
        
        Ok(())
    }
    
    /// Final cleanup phase like C++
    fn final_cleanup(&self, visit_order: &[usize], vectors: &[Vec<f32>]) -> Result<()> {
        println!("   🧹 Starting final cleanup...");
        
        #[cfg(feature = "parallel")]
        {
            visit_order
                .par_chunks(2048)
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
        
        #[cfg(not(feature = "parallel"))]
        {
            for &node in visit_order {
                let needs_pruning = {
                    let graph = self.graph.read();
                    graph[node].len() > self.params.max_degree
                };
                
                if needs_pruning {
                    self.cleanup_node(node, vectors)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Cleanup a single node
    fn cleanup_node(&self, node: usize, vectors: &[Vec<f32>]) -> Result<()> {
        let neighbors = {
            let graph = self.graph.read();
            graph[node].clone()
        };
        
        let mut pool = Vec::new();
        for &neighbor in &neighbors {
            let dist = self.distance_fn.distance(&vectors[node], &vectors[neighbor])?;
            pool.push(Neighbor { id: neighbor, distance: dist });
        }
        
        pool.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        let pruned = self.robust_prune(node, pool, vectors)?;
        
        let mut graph = self.graph.write();
        graph[node] = pruned;
        
        Ok(())
    }
    
    /// Calculate entry point (medoid)
    fn calculate_entry_point(&self, vectors: &[Vec<f32>]) -> Result<usize> {
        // Use our optimized NEON medoid calculation
        self.find_medoid_centroid(vectors)
    }
    
    /// Find medoid using centroid method (O(n))
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