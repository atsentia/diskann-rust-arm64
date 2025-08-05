//! Dynamic index operations (insert, delete, consolidate)
//!
//! This module provides support for dynamic updates to the index including
//! insertion of new vectors, deletion with lazy marking, and consolidation.

use crate::{Distance, Result, Error};
use crate::graph::VamanaGraph;
use crate::distance::create_distance_function;
use crate::labels::{LabelSet, LabelIndex};
use parking_lot::RwLock;
use std::sync::Arc;
use std::collections::{HashSet, HashMap};
use crossbeam_channel::{bounded, Sender, Receiver};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};

/// Dynamic index that supports insertions and deletions
pub struct DynamicIndex {
    /// Base vectors (includes deleted)
    vectors: Arc<RwLock<Vec<Option<Vec<f32>>>>>,
    /// Graph structure
    graph: Arc<RwLock<VamanaGraph>>,
    /// Label index
    labels: Arc<RwLock<LabelIndex>>,
    /// Deleted vector IDs
    deleted: Arc<RwLock<HashSet<usize>>>,
    /// Free slots from deletions
    free_slots: Arc<RwLock<Vec<usize>>>,
    /// Configuration
    dimension: usize,
    metric: Distance,
    max_degree: usize,
    search_list_size: usize,
    alpha: f32,
    /// Statistics
    num_vectors: AtomicUsize,
    num_deleted: AtomicUsize,
    /// Consolidation threshold (percentage of deleted vectors)
    consolidation_threshold: f32,
}

impl DynamicIndex {
    /// Create a new dynamic index
    pub fn new(
        dimension: usize,
        metric: Distance,
        max_degree: usize,
        search_list_size: usize,
        alpha: f32,
    ) -> Self {
        Self {
            vectors: Arc::new(RwLock::new(Vec::new())),
            graph: Arc::new(RwLock::new(VamanaGraph::new(
                0,
                dimension,
                metric,
                max_degree,
                search_list_size,
                alpha,
            ))),
            labels: Arc::new(RwLock::new(LabelIndex::new(0))),
            deleted: Arc::new(RwLock::new(HashSet::new())),
            free_slots: Arc::new(RwLock::new(Vec::new())),
            dimension,
            metric,
            max_degree,
            search_list_size,
            alpha,
            num_vectors: AtomicUsize::new(0),
            num_deleted: AtomicUsize::new(0),
            consolidation_threshold: 0.2, // Consolidate when 20% deleted
        }
    }
    
    /// Insert a new vector
    pub fn insert(&self, vector: Vec<f32>, labels: Vec<u32>) -> Result<usize> {
        if vector.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            }.into());
        }
        
        // Get or allocate ID
        let id = {
            let mut free_slots = self.free_slots.write();
            if let Some(id) = free_slots.pop() {
                id
            } else {
                let mut vectors = self.vectors.write();
                let id = vectors.len();
                vectors.push(None);
                id
            }
        };
        
        // Insert vector
        {
            let mut vectors = self.vectors.write();
            vectors[id] = Some(vector.clone());
        }
        
        // Update labels
        {
            let mut label_index = self.labels.write();
            label_index.set_labels(id, labels);
        }
        
        // Update graph
        self.update_graph_for_insert(id, &vector)?;
        
        // Update stats
        self.num_vectors.fetch_add(1, Ordering::Relaxed);
        
        Ok(id)
    }
    
    /// Delete a vector (lazy deletion)
    pub fn delete(&self, id: usize) -> Result<()> {
        // Check if valid
        {
            let vectors = self.vectors.read();
            if id >= vectors.len() || vectors[id].is_none() {
                return Err(Error::InvalidParameter(format!("Invalid vector ID: {}", id)).into());
            }
        }
        
        // Check if already deleted
        {
            let deleted = self.deleted.read();
            if deleted.contains(&id) {
                return Ok(()); // Already deleted
            }
        }
        
        // Mark as deleted
        {
            let mut deleted = self.deleted.write();
            deleted.insert(id);
        }
        
        // Clear vector data (keep slot)
        {
            let mut vectors = self.vectors.write();
            vectors[id] = None;
        }
        
        // Clear labels
        {
            let mut labels = self.labels.write();
            labels.set_labels(id, vec![]);
        }
        
        // Add to free slots
        {
            let mut free_slots = self.free_slots.write();
            free_slots.push(id);
        }
        
        // Update graph edges (remove connections to deleted node)
        self.update_graph_for_delete(id)?;
        
        // Update stats
        self.num_deleted.fetch_add(1, Ordering::Relaxed);
        self.num_vectors.fetch_sub(1, Ordering::Relaxed);
        
        // Check if consolidation needed
        self.maybe_consolidate()?;
        
        Ok(())
    }
    
    /// Search for nearest neighbors (excludes deleted vectors)
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            }.into());
        }
        
        let vectors = self.vectors.read();
        let deleted = self.deleted.read();
        
        // Search using graph with dynamic vectors
        let graph = self.graph.read();
        let mut results = graph.search_dynamic(query, k + deleted.len(), &*vectors)?;
        
        // Filter out deleted vectors from results
        results.retain(|(id, _)| !deleted.contains(id));
        results.truncate(k);
        
        Ok(results)
    }
    
    /// Consolidate the index to remove deleted vectors
    pub fn consolidate(&self) -> Result<()> {
        let num_deleted = self.num_deleted.load(Ordering::Relaxed);
        if num_deleted == 0 {
            return Ok(()); // Nothing to consolidate
        }
        
        // Create mapping from old to new IDs
        let (new_vectors, id_mapping) = {
            let vectors = self.vectors.read();
            let deleted = self.deleted.read();
            
            let mut new_vectors = Vec::new();
            let mut id_mapping = HashMap::new();
            
            for (old_id, vector) in vectors.iter().enumerate() {
                if !deleted.contains(&old_id) {
                    if let Some(v) = vector {
                        let new_id = new_vectors.len();
                        id_mapping.insert(old_id, new_id);
                        new_vectors.push(v.clone());
                    }
                }
            }
            
            (new_vectors, id_mapping)
        };
        
        // Rebuild graph with new IDs
        let mut new_graph = VamanaGraph::new(
            new_vectors.len(),
            self.dimension,
            self.metric,
            self.max_degree,
            self.search_list_size,
            self.alpha,
        );
        new_graph.build(&new_vectors)?;
        
        // Update label index with new IDs
        let mut new_labels = LabelIndex::new(new_vectors.len());
        {
            let labels = self.labels.read();
            for (old_id, new_id) in &id_mapping {
                if let Some(label_set) = labels.get_labels(*old_id) {
                    new_labels.set_labels(*new_id, label_set.labels().to_vec());
                }
            }
        }
        
        // Atomically update all structures
        {
            let mut vectors = self.vectors.write();
            let mut graph = self.graph.write();
            let mut labels = self.labels.write();
            let mut deleted = self.deleted.write();
            let mut free_slots = self.free_slots.write();
            
            *vectors = new_vectors.into_iter().map(Some).collect();
            *graph = new_graph;
            *labels = new_labels;
            deleted.clear();
            free_slots.clear();
        }
        
        // Update stats
        self.num_deleted.store(0, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Get index statistics
    pub fn stats(&self) -> DynamicIndexStats {
        DynamicIndexStats {
            num_vectors: self.num_vectors.load(Ordering::Relaxed),
            num_deleted: self.num_deleted.load(Ordering::Relaxed),
            dimension: self.dimension,
            metric: self.metric,
            fragmentation: self.fragmentation_ratio(),
        }
    }
    
    /// Update graph for insertion
    fn update_graph_for_insert(&self, id: usize, vector: &[f32]) -> Result<()> {
        let vectors = self.vectors.read();
        let graph = self.graph.read();
        
        // Use the VamanaGraph's insert_single method
        graph.insert_single(id, vector, &*vectors)?;
        
        Ok(())
    }
    
    /// Update graph for deletion
    fn update_graph_for_delete(&self, id: usize) -> Result<()> {
        let graph = self.graph.read();
        
        // Use the VamanaGraph's delete_vertex method
        graph.delete_vertex(id)?;
        
        Ok(())
    }
    
    /// Check if consolidation is needed
    fn maybe_consolidate(&self) -> Result<()> {
        let fragmentation = self.fragmentation_ratio();
        if fragmentation > self.consolidation_threshold {
            self.consolidate()?;
        }
        Ok(())
    }
    
    /// Calculate fragmentation ratio
    fn fragmentation_ratio(&self) -> f32 {
        let total = self.num_vectors.load(Ordering::Relaxed) + self.num_deleted.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            self.num_deleted.load(Ordering::Relaxed) as f32 / total as f32
        }
    }
}

/// Dynamic index statistics
#[derive(Debug)]
pub struct DynamicIndexStats {
    pub num_vectors: usize,
    pub num_deleted: usize,
    pub dimension: usize,
    pub metric: Distance,
    pub fragmentation: f32,
}

/// Streaming index for continuous updates
pub struct StreamingIndex {
    /// Dynamic index
    index: Arc<DynamicIndex>,
    /// Update channel
    update_tx: Sender<UpdateOp>,
    update_rx: Receiver<UpdateOp>,
    /// Background thread handle
    worker_thread: Option<std::thread::JoinHandle<()>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// Update operation for streaming
enum UpdateOp {
    Insert { vector: Vec<f32>, labels: Vec<u32>, callback: Sender<Result<usize>> },
    Delete { id: usize, callback: Sender<Result<()>> },
    Consolidate { callback: Sender<Result<()>> },
}

impl StreamingIndex {
    /// Create a new streaming index
    pub fn new(
        dimension: usize,
        metric: Distance,
        max_degree: usize,
        search_list_size: usize,
        alpha: f32,
    ) -> Self {
        let index = Arc::new(DynamicIndex::new(
            dimension,
            metric,
            max_degree,
            search_list_size,
            alpha,
        ));
        
        let (update_tx, update_rx) = bounded(1000);
        let shutdown = Arc::new(AtomicBool::new(false));
        
        let mut streaming = Self {
            index: index.clone(),
            update_tx,
            update_rx,
            worker_thread: None,
            shutdown: shutdown.clone(),
        };
        
        // Start background worker
        let worker_index = index.clone();
        let worker_rx = streaming.update_rx.clone();
        let worker_shutdown = shutdown.clone();
        
        let handle = std::thread::spawn(move || {
            while !worker_shutdown.load(Ordering::Relaxed) {
                match worker_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(op) => {
                        match op {
                            UpdateOp::Insert { vector, labels, callback } => {
                                let result = worker_index.insert(vector, labels);
                                let _ = callback.send(result);
                            }
                            UpdateOp::Delete { id, callback } => {
                                let result = worker_index.delete(id);
                                let _ = callback.send(result);
                            }
                            UpdateOp::Consolidate { callback } => {
                                let result = worker_index.consolidate();
                                let _ = callback.send(result);
                            }
                        }
                    }
                    Err(_) => {} // Timeout, continue
                }
            }
        });
        
        streaming.worker_thread = Some(handle);
        streaming
    }
    
    /// Insert a vector asynchronously
    pub async fn insert_async(&self, vector: Vec<f32>, labels: Vec<u32>) -> Result<usize> {
        let (callback_tx, callback_rx) = bounded(1);
        
        self.update_tx.send(UpdateOp::Insert {
            vector,
            labels,
            callback: callback_tx,
        }).map_err(|_| anyhow::anyhow!("Update channel closed"))?;
        
        callback_rx.recv()
            .map_err(|_| anyhow::anyhow!("Callback channel closed"))?
    }
    
    /// Delete a vector asynchronously
    pub async fn delete_async(&self, id: usize) -> Result<()> {
        let (callback_tx, callback_rx) = bounded(1);
        
        self.update_tx.send(UpdateOp::Delete {
            id,
            callback: callback_tx,
        }).map_err(|_| anyhow::anyhow!("Update channel closed"))?;
        
        callback_rx.recv()
            .map_err(|_| anyhow::anyhow!("Callback channel closed"))?
    }
    
    /// Trigger consolidation asynchronously
    pub async fn consolidate_async(&self) -> Result<()> {
        let (callback_tx, callback_rx) = bounded(1);
        
        self.update_tx.send(UpdateOp::Consolidate {
            callback: callback_tx,
        }).map_err(|_| anyhow::anyhow!("Update channel closed"))?;
        
        callback_rx.recv()
            .map_err(|_| anyhow::anyhow!("Callback channel closed"))?
    }
    
    /// Search (immediate, not queued)
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        self.index.search(query, k)
    }
    
    /// Shutdown the streaming index
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.worker_thread.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dynamic_insert_delete() {
        let index = DynamicIndex::new(4, Distance::L2, 16, 32, 1.2);
        
        // Insert vectors
        let id1 = index.insert(vec![1.0, 0.0, 0.0, 0.0], vec![1]).unwrap();
        let id2 = index.insert(vec![0.0, 1.0, 0.0, 0.0], vec![2]).unwrap();
        let id3 = index.insert(vec![0.0, 0.0, 1.0, 0.0], vec![3]).unwrap();
        
        assert_eq!(index.stats().num_vectors, 3);
        assert_eq!(index.stats().num_deleted, 0);
        
        // Delete a vector
        index.delete(id2).unwrap();
        
        assert_eq!(index.stats().num_vectors, 2);
        assert_eq!(index.stats().num_deleted, 1);
        
        // Search should not return deleted vector
        let results = index.search(&[0.0, 1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 2);
        assert!(!results.iter().any(|(id, _)| *id == id2));
    }
    
    #[test]
    fn test_consolidation() {
        let index = DynamicIndex::new(2, Distance::L2, 16, 32, 1.2);
        
        // Insert and delete many vectors
        let mut ids = Vec::new();
        for i in 0..10 {
            let id = index.insert(vec![i as f32, 0.0], vec![]).unwrap();
            ids.push(id);
        }
        
        // Delete half
        for i in 0..5 {
            index.delete(ids[i]).unwrap();
        }
        
        assert_eq!(index.stats().num_deleted, 5);
        
        // Consolidate
        index.consolidate().unwrap();
        
        assert_eq!(index.stats().num_vectors, 5);
        assert_eq!(index.stats().num_deleted, 0);
        assert_eq!(index.stats().fragmentation, 0.0);
    }
    
    #[tokio::test]
    async fn test_streaming_index() {
        let index = StreamingIndex::new(3, Distance::L2, 16, 32, 1.2);
        
        // Async insert
        let id = index.insert_async(vec![1.0, 2.0, 3.0], vec![10]).await.unwrap();
        
        // Search
        let results = index.search(&[1.0, 2.0, 3.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        
        // Async delete
        index.delete_async(id).await.unwrap();
        
        // Verify deletion
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let results = index.search(&[1.0, 2.0, 3.0], 1).unwrap();
        assert_eq!(results.len(), 0);
        
        index.shutdown();
    }
}