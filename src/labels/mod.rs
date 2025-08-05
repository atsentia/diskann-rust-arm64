//! Label and filter support for DiskANN
//!
//! This module provides label-based filtering capabilities for search operations.

use crate::Result;
use hashbrown::{HashMap, HashSet};

/// Label type (32-bit for compatibility with C++)
pub type Label = u32;

/// Special label values
pub const UNIVERSAL_LABEL: Label = 0;
pub const INVALID_LABEL: Label = u32::MAX;

/// Label set for a single vector
#[derive(Debug, Clone, Default)]
pub struct LabelSet {
    labels: Vec<Label>,
}

impl LabelSet {
    /// Create a new empty label set
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create from a vector of labels
    pub fn from_vec(labels: Vec<Label>) -> Self {
        let mut label_set = Self { labels };
        label_set.labels.sort_unstable();
        label_set.labels.dedup();
        label_set
    }
    
    /// Add a label
    pub fn add(&mut self, label: Label) {
        if !self.labels.contains(&label) {
            self.labels.push(label);
            self.labels.sort_unstable();
        }
    }
    
    /// Remove a label
    pub fn remove(&mut self, label: Label) {
        self.labels.retain(|&l| l != label);
    }
    
    /// Check if contains a label
    pub fn contains(&self, label: Label) -> bool {
        self.labels.binary_search(&label).is_ok()
    }
    
    /// Check if matches any of the filter labels
    pub fn matches_any(&self, filter: &[Label]) -> bool {
        for &label in &self.labels {
            if label == UNIVERSAL_LABEL || filter.binary_search(&label).is_ok() {
                return true;
            }
        }
        false
    }
    
    /// Check if matches all of the filter labels
    pub fn matches_all(&self, filter: &[Label]) -> bool {
        if self.contains(UNIVERSAL_LABEL) {
            return true;
        }
        
        for &label in filter {
            if !self.contains(label) {
                return false;
            }
        }
        true
    }
    
    /// Get the labels as a slice
    pub fn labels(&self) -> &[Label] {
        &self.labels
    }
    
    /// Get the number of labels
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
}

/// Label index for efficient filtering
pub struct LabelIndex {
    /// Mapping from vector ID to its labels
    vector_labels: Vec<LabelSet>,
    /// Inverted index: label -> vector IDs
    label_vectors: HashMap<Label, Vec<usize>>,
    /// Set of all unique labels
    unique_labels: HashSet<Label>,
}

impl LabelIndex {
    /// Create a new label index
    pub fn new(num_vectors: usize) -> Self {
        Self {
            vector_labels: vec![LabelSet::new(); num_vectors],
            label_vectors: HashMap::new(),
            unique_labels: HashSet::new(),
        }
    }
    
    /// Build from vector labels
    pub fn build(labels_per_vector: Vec<Vec<Label>>) -> Self {
        let num_vectors = labels_per_vector.len();
        let mut index = Self::new(num_vectors);
        
        for (vector_id, labels) in labels_per_vector.into_iter().enumerate() {
            index.set_labels(vector_id, labels);
        }
        
        index
    }
    
    /// Set labels for a vector
    pub fn set_labels(&mut self, vector_id: usize, labels: Vec<Label>) {
        if vector_id >= self.vector_labels.len() {
            self.vector_labels.resize(vector_id + 1, LabelSet::new());
        }
        
        // Remove old labels from inverted index
        for &label in self.vector_labels[vector_id].labels() {
            if let Some(vectors) = self.label_vectors.get_mut(&label) {
                vectors.retain(|&id| id != vector_id);
                if vectors.is_empty() {
                    self.label_vectors.remove(&label);
                    self.unique_labels.remove(&label);
                }
            }
        }
        
        // Set new labels
        self.vector_labels[vector_id] = LabelSet::from_vec(labels);
        
        // Update inverted index
        for &label in self.vector_labels[vector_id].labels() {
            self.label_vectors
                .entry(label)
                .or_insert_with(Vec::new)
                .push(vector_id);
            self.unique_labels.insert(label);
        }
    }
    
    /// Get labels for a vector
    pub fn get_labels(&self, vector_id: usize) -> Option<Vec<Label>> {
        self.vector_labels.get(vector_id).map(|ls| ls.labels().to_vec())
    }
    
    /// Get label set for a vector
    pub fn get_label_set(&self, vector_id: usize) -> Option<&LabelSet> {
        self.vector_labels.get(vector_id)
    }
    
    /// Get vectors with a specific label
    pub fn get_vectors_with_label(&self, label: Label) -> &[usize] {
        self.label_vectors
            .get(&label)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    
    /// Check if vector matches filter
    pub fn matches_filter(&self, vector_id: usize, filter: &[Label]) -> bool {
        if filter.is_empty() {
            return true; // No filter means match all
        }
        
        if let Some(label_set) = self.vector_labels.get(vector_id) {
            label_set.matches_any(filter)
        } else {
            false
        }
    }
    
    /// Get all unique labels
    pub fn unique_labels(&self) -> Vec<Label> {
        let mut labels: Vec<Label> = self.unique_labels.iter().copied().collect();
        labels.sort_unstable();
        labels
    }
    
    /// Get candidate vectors that match the given filter
    pub fn get_candidates(&self, filter: &LabelFilter) -> Result<HashSet<usize>> {
        let mut candidates = HashSet::new();
        
        match filter {
            LabelFilter::Any => {
                // Return all vectors
                for i in 0..self.vector_labels.len() {
                    candidates.insert(i);
                }
            }
            LabelFilter::AnyOf(labels) => {
                // Union of all vectors with any of these labels
                for &label in labels {
                    for &vector_id in self.get_vectors_with_label(label) {
                        candidates.insert(vector_id);
                    }
                }
                
                // Also include vectors with universal label
                for &vector_id in self.get_vectors_with_label(UNIVERSAL_LABEL) {
                    candidates.insert(vector_id);
                }
            }
            LabelFilter::AllOf(labels) => {
                if labels.is_empty() {
                    // If no labels specified, return all vectors
                    for i in 0..self.vector_labels.len() {
                        candidates.insert(i);
                    }
                } else {
                    // Intersection of vectors with all these labels
                    let mut first = true;
                    for &label in labels {
                        let label_vectors: HashSet<usize> = self.get_vectors_with_label(label)
                            .iter()
                            .copied()
                            .collect();
                        
                        if first {
                            candidates = label_vectors;
                            first = false;
                        } else {
                            candidates = candidates.intersection(&label_vectors).copied().collect();
                        }
                    }
                    
                    // Also include vectors with universal label
                    for &vector_id in self.get_vectors_with_label(UNIVERSAL_LABEL) {
                        candidates.insert(vector_id);
                    }
                }
            }
            LabelFilter::Exact(labels) => {
                // Find vectors with exactly these labels
                for (vector_id, label_set) in self.vector_labels.iter().enumerate() {
                    if filter.matches(label_set) {
                        candidates.insert(vector_id);
                    }
                }
            }
        }
        
        Ok(candidates)
    }
    
    /// Get label statistics
    pub fn stats(&self) -> LabelStats {
        let mut label_counts: HashMap<Label, usize> = HashMap::new();
        let mut max_labels_per_vector = 0;
        let mut total_labels = 0;
        
        for label_set in &self.vector_labels {
            max_labels_per_vector = max_labels_per_vector.max(label_set.len());
            total_labels += label_set.len();
            
            for &label in label_set.labels() {
                *label_counts.entry(label).or_insert(0) += 1;
            }
        }
        
        let avg_labels_per_vector = if self.vector_labels.is_empty() {
            0.0
        } else {
            total_labels as f32 / self.vector_labels.len() as f32
        };
        
        LabelStats {
            num_vectors: self.vector_labels.len(),
            num_unique_labels: self.unique_labels.len(),
            max_labels_per_vector,
            avg_labels_per_vector,
            label_distribution: label_counts,
        }
    }
}

/// Label index statistics
#[derive(Debug)]
pub struct LabelStats {
    pub num_vectors: usize,
    pub num_unique_labels: usize,
    pub max_labels_per_vector: usize,
    pub avg_labels_per_vector: f32,
    pub label_distribution: HashMap<Label, usize>,
}

/// Label filter for search operations
#[derive(Debug, Clone)]
pub enum LabelFilter {
    /// Match any vector (no filtering)
    Any,
    /// Match vectors with any of these labels (OR operation)
    AnyOf(Vec<Label>),
    /// Match vectors with all of these labels (AND operation)
    AllOf(Vec<Label>),
    /// Match vectors with exactly these labels
    Exact(Vec<Label>),
}

impl LabelFilter {
    /// Create a filter that matches any of the given labels
    pub fn any_of(labels: Vec<Label>) -> Self {
        let mut sorted_labels = labels;
        sorted_labels.sort_unstable();
        sorted_labels.dedup();
        LabelFilter::AnyOf(sorted_labels)
    }
    
    /// Create a filter that matches all of the given labels
    pub fn all_of(labels: Vec<Label>) -> Self {
        let mut sorted_labels = labels;
        sorted_labels.sort_unstable();
        sorted_labels.dedup();
        LabelFilter::AllOf(sorted_labels)
    }
    
    /// Create a filter that matches exactly the given labels
    pub fn exact(labels: Vec<Label>) -> Self {
        let mut sorted_labels = labels;
        sorted_labels.sort_unstable();
        sorted_labels.dedup();
        LabelFilter::Exact(sorted_labels)
    }
    
    /// Check if this filter matches a label set
    pub fn matches(&self, label_set: &LabelSet) -> bool {
        match self {
            LabelFilter::Any => true,
            LabelFilter::AnyOf(labels) => label_set.matches_any(labels),
            LabelFilter::AllOf(labels) => label_set.matches_all(labels),
            LabelFilter::Exact(labels) => {
                let set_labels = label_set.labels();
                set_labels.len() == labels.len() && 
                set_labels.iter().zip(labels.iter()).all(|(a, b)| a == b)
            }
        }
    }
}

/// Filter parameters for search (deprecated - use LabelFilter instead)
#[derive(Debug, Clone)]
pub struct FilterParams {
    /// Labels to filter by (OR operation)
    pub labels: Vec<Label>,
    /// Whether to use AND operation instead of OR
    pub match_all: bool,
}

impl FilterParams {
    /// Create new filter parameters
    pub fn new(labels: Vec<Label>) -> Self {
        let mut params = Self {
            labels,
            match_all: false,
        };
        params.labels.sort_unstable();
        params.labels.dedup();
        params
    }
    
    /// Set to match all labels (AND operation)
    pub fn match_all(mut self) -> Self {
        self.match_all = true;
        self
    }
    
    /// Check if empty (no filtering)
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_label_set() {
        let mut label_set = LabelSet::new();
        assert!(label_set.is_empty());
        
        label_set.add(5);
        label_set.add(3);
        label_set.add(5); // Duplicate
        label_set.add(7);
        
        assert_eq!(label_set.len(), 3);
        assert!(label_set.contains(3));
        assert!(label_set.contains(5));
        assert!(label_set.contains(7));
        assert!(!label_set.contains(1));
        
        // Test matching
        assert!(label_set.matches_any(&[1, 3, 9]));
        assert!(!label_set.matches_any(&[1, 2, 4]));
        assert!(label_set.matches_all(&[3, 5]));
        assert!(!label_set.matches_all(&[3, 5, 9]));
    }
    
    #[test]
    fn test_label_index() {
        let labels = vec![
            vec![1, 2],      // Vector 0
            vec![2, 3],      // Vector 1
            vec![1],         // Vector 2
            vec![UNIVERSAL_LABEL], // Vector 3
        ];
        
        let index = LabelIndex::build(labels);
        
        // Test vector labels
        assert_eq!(index.get_labels(0).unwrap().labels(), &[1, 2]);
        assert_eq!(index.get_labels(3).unwrap().labels(), &[UNIVERSAL_LABEL]);
        
        // Test inverted index
        assert_eq!(index.get_vectors_with_label(1), &[0, 2]);
        assert_eq!(index.get_vectors_with_label(2), &[0, 1]);
        assert_eq!(index.get_vectors_with_label(3), &[1]);
        
        // Test filtering
        assert!(index.matches_filter(0, &[1]));
        assert!(index.matches_filter(0, &[2]));
        assert!(!index.matches_filter(0, &[3]));
        assert!(index.matches_filter(3, &[99])); // Universal label matches any
        
        // Test stats
        let stats = index.stats();
        assert_eq!(stats.num_vectors, 4);
        assert_eq!(stats.num_unique_labels, 4); // Including universal label
        assert_eq!(stats.max_labels_per_vector, 2);
    }
}