//! Graph pruning algorithms for DiskANN
//!
//! This module implements the RobustPrune algorithm and related pruning strategies.

use crate::Result;
use std::collections::HashSet;

/// RobustPrune algorithm for edge selection
pub fn robust_prune<F>(
    vertex_id: usize,
    candidates: &[usize],
    vectors: &[Vec<f32>],
    max_degree: usize,
    alpha: f32,
    mut distance_fn: F,
) -> Result<Vec<usize>>
where
    F: FnMut(&[f32], &[f32]) -> Result<f32>,
{
    if candidates.len() <= max_degree {
        return Ok(candidates.to_vec());
    }
    
    // Calculate distances from vertex to all candidates
    let vertex = &vectors[vertex_id];
    let mut neighbors: Vec<(usize, f32)> = candidates
        .iter()
        .filter(|&&id| id != vertex_id)
        .map(|&id| {
            let dist = distance_fn(vertex, &vectors[id]).unwrap();
            (id, dist)
        })
        .collect();
    
    // Sort by distance
    neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    // RobustPrune selection
    let mut pruned = Vec::with_capacity(max_degree);
    let mut pruned_set = HashSet::new();
    
    for (candidate_id, candidate_dist) in neighbors {
        if pruned.len() >= max_degree {
            break;
        }
        
        // Check if this candidate should be pruned
        let mut should_prune = false;
        
        for &selected_id in &pruned {
            let dist_to_selected = distance_fn(
                &vectors[candidate_id][..],
                &vectors[selected_id][..]
            )?;
            
            // Prune if candidate is closer to a selected neighbor than to vertex
            if dist_to_selected < candidate_dist * alpha {
                should_prune = true;
                break;
            }
        }
        
        if !should_prune {
            pruned.push(candidate_id);
            pruned_set.insert(candidate_id);
        }
    }
    
    // If we pruned too aggressively, add back some neighbors
    if pruned.len() < max_degree / 2 {
        for (id, _) in neighbors {
            if !pruned_set.contains(&id) && pruned.len() < max_degree {
                pruned.push(id);
                pruned_set.insert(id);
            }
        }
    }
    
    Ok(pruned)
}

/// Prune edges while maintaining graph connectivity
pub fn prune_with_connectivity<F>(
    graph: &mut Vec<Vec<usize>>,
    vertex_id: usize,
    vectors: &[Vec<f32>],
    max_degree: usize,
    alpha: f32,
    distance_fn: F,
) -> Result<()>
where
    F: FnMut(&[f32], &[f32]) -> Result<f32>,
{
    let candidates = graph[vertex_id].clone();
    
    if candidates.len() <= max_degree {
        return Ok(());
    }
    
    // Get pruned neighbors
    let pruned = robust_prune(vertex_id, &candidates, vectors, max_degree, alpha, distance_fn)?;
    
    // Track removed edges
    let pruned_set: HashSet<usize> = pruned.iter().cloned().collect();
    let removed: Vec<usize> = candidates
        .into_iter()
        .filter(|id| !pruned_set.contains(id))
        .collect();
    
    // Update forward edges
    graph[vertex_id] = pruned;
    
    // Update reverse edges
    for neighbor_id in removed {
        graph[neighbor_id].retain(|&id| id != vertex_id);
    }
    
    Ok(())
}

/// Batch pruning for multiple vertices
pub fn batch_prune<F>(
    graph: &mut Vec<Vec<usize>>,
    vertices: &[usize],
    vectors: &[Vec<f32>],
    max_degree: usize,
    alpha: f32,
    mut distance_fn: F,
) -> Result<()>
where
    F: FnMut(&[f32], &[f32]) -> Result<f32> + Clone,
{
    for &vertex_id in vertices {
        prune_with_connectivity(
            graph,
            vertex_id,
            vectors,
            max_degree,
            alpha,
            distance_fn.clone(),
        )?;
    }
    
    Ok(())
}

/// Calculate the occlude factor for a candidate
pub fn occlude_factor<F>(
    candidate_id: usize,
    vertex_id: usize,
    selected: &[usize],
    vectors: &[Vec<f32>],
    mut distance_fn: F,
) -> Result<f32>
where
    F: FnMut(&[f32], &[f32]) -> Result<f32>,
{
    let dist_to_vertex = distance_fn(&vectors[candidate_id], &vectors[vertex_id])?;
    
    let mut min_dist_to_selected = f32::MAX;
    for &selected_id in selected {
        let dist = distance_fn(&vectors[candidate_id], &vectors[selected_id])?;
        min_dist_to_selected = min_dist_to_selected.min(dist);
    }
    
    Ok(min_dist_to_selected / dist_to_vertex)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{Distance, create_distance_function};
    use crate::utils::generate_random_vectors;
    
    #[test]
    fn test_robust_prune() {
        let vectors = generate_random_vectors(10, 4);
        let candidates: Vec<usize> = (1..10).collect();
        let distance_fn = create_distance_function(Distance::L2, 4);
        
        let pruned = robust_prune(
            0,
            &candidates,
            &vectors,
            5,
            1.2,
            |a, b| distance_fn.distance(a, b),
        ).unwrap();
        
        assert!(pruned.len() <= 5);
        assert!(pruned.len() >= 2); // Should keep at least some neighbors
    }
    
    #[test]
    fn test_occlude_factor() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        
        let distance_fn = create_distance_function(Distance::L2, 2);
        let factor = occlude_factor(
            3, // candidate at (1,1)
            0, // vertex at (0,0)
            &[1, 2], // selected at (1,0) and (0,1)
            &vectors,
            |a, b| distance_fn.distance(a, b),
        ).unwrap();
        
        assert!(factor < 1.0); // Should be occluded by selected neighbors
    }
}