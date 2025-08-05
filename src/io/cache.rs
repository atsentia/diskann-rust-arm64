//! Caching utilities for disk-based operations
//!
//! This module provides caching mechanisms for DiskANN.

use parking_lot::RwLock;
use hashbrown::HashMap;
use std::sync::Arc;

/// Cache entry
pub struct CacheEntry<T> {
    pub data: T,
    pub access_count: usize,
}

/// Generic LRU cache
pub struct LruCache<K, V> {
    capacity: usize,
    cache: RwLock<HashMap<K, CacheEntry<V>>>,
    access_counter: Arc<RwLock<usize>>,
}

impl<K: Eq + std::hash::Hash + Clone, V: Clone> LruCache<K, V> {
    /// Create a new LRU cache
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: RwLock::new(HashMap::new()),
            access_counter: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write();
        if let Some(entry) = cache.get_mut(key) {
            let mut counter = self.access_counter.write();
            *counter += 1;
            entry.access_count = *counter;
            Some(entry.data.clone())
        } else {
            None
        }
    }
    
    /// Put a value into the cache
    pub fn put(&self, key: K, value: V) {
        let mut cache = self.cache.write();
        let mut counter = self.access_counter.write();
        
        if cache.len() >= self.capacity && !cache.contains_key(&key) {
            // Evict least recently used
            if let Some((lru_key, _)) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.access_count)
                .map(|(k, v)| (k.clone(), v.access_count))
            {
                cache.remove(&lru_key);
            }
        }
        
        *counter += 1;
        cache.insert(key, CacheEntry {
            data: value,
            access_count: *counter,
        });
    }
    
    /// Check if a key exists in the cache
    pub fn contains(&self, key: &K) -> bool {
        self.cache.read().contains_key(key)
    }
    
    /// Clear the cache
    pub fn clear(&self) {
        self.cache.write().clear();
    }
    
    /// Get the current size of the cache
    pub fn size(&self) -> usize {
        self.cache.read().len()
    }
}