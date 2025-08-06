//! Natural number containers for high-performance integer key operations
//!
//! This module provides specialized data structures optimized for natural number keys,
//! similar to the `natural_number_map.h` from the C++ DiskANN implementation.

use std::ops::{Index, IndexMut};

/// High-performance map for natural number keys using direct indexing
#[derive(Debug, Clone)]
pub struct NaturalNumberMap<T> {
    data: Vec<Option<T>>,
    size: usize,
}

impl<T> NaturalNumberMap<T> {
    /// Create a new natural number map with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: {
                let mut vec = Vec::with_capacity(capacity);
                vec.resize_with(capacity, || None);
                vec
            },
            size: 0,
        }
    }

    /// Create a new natural number map
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    /// Insert a value at the given natural number key
    pub fn insert(&mut self, key: usize, value: T) -> Option<T> {
        self.ensure_capacity(key + 1);
        
        let old_value = std::mem::replace(&mut self.data[key], Some(value));
        if old_value.is_none() {
            self.size += 1;
        }
        old_value
    }

    /// Get a value by natural number key
    pub fn get(&self, key: usize) -> Option<&T> {
        self.data.get(key).and_then(|v| v.as_ref())
    }

    /// Get a mutable reference to a value by natural number key
    pub fn get_mut(&mut self, key: usize) -> Option<&mut T> {
        self.data.get_mut(key).and_then(|v| v.as_mut())
    }

    /// Remove a value at the given key
    pub fn remove(&mut self, key: usize) -> Option<T> {
        if key < self.data.len() {
            let old_value = std::mem::take(&mut self.data[key]);
            if old_value.is_some() {
                self.size -= 1;
            }
            old_value
        } else {
            None
        }
    }

    /// Check if a key exists in the map
    pub fn contains_key(&self, key: usize) -> bool {
        key < self.data.len() && self.data[key].is_some()
    }

    /// Get the number of elements in the map
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the map is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get the current capacity
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.data.clear();
        self.size = 0;
    }

    /// Iterate over key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|val| (i, val)))
    }

    /// Iterate over mutable key-value pairs
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut T)> {
        self.data
            .iter_mut()
            .enumerate()
            .filter_map(|(i, v)| v.as_mut().map(|val| (i, val)))
    }

    /// Iterate over keys
    pub fn keys(&self) -> impl Iterator<Item = usize> + '_ {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if v.is_some() { Some(i) } else { None })
    }

    /// Iterate over values
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.data.iter().filter_map(|v| v.as_ref())
    }

    /// Iterate over mutable values
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().filter_map(|v| v.as_mut())
    }

    /// Ensure the capacity is at least min_capacity
    fn ensure_capacity(&mut self, min_capacity: usize) {
        if self.data.len() < min_capacity {
            let new_capacity = std::cmp::max(min_capacity, self.data.len() * 2);
            self.data.resize_with(new_capacity, || None);
        }
    }

    /// Shrink the capacity to fit the current size
    pub fn shrink_to_fit(&mut self) {
        // Find the highest key in use
        let max_key = self.data
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, v)| if v.is_some() { Some(i + 1) } else { None })
            .unwrap_or(0);
        
        if max_key < self.data.len() {
            self.data.truncate(max_key);
        }
    }

    /// Reserve additional capacity
    pub fn reserve(&mut self, additional: usize) {
        let new_capacity = self.size + additional;
        if new_capacity > self.data.len() {
            self.data.resize_with(new_capacity, || None);
        }
    }
}

impl<T> Default for NaturalNumberMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Index<usize> for NaturalNumberMap<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Key not found in NaturalNumberMap")
    }
}

impl<T> IndexMut<usize> for NaturalNumberMap<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("Key not found in NaturalNumberMap")
    }
}

impl<T> FromIterator<(usize, T)> for NaturalNumberMap<T> {
    fn from_iter<I: IntoIterator<Item = (usize, T)>>(iter: I) -> Self {
        let mut map = Self::new();
        for (key, value) in iter {
            map.insert(key, value);
        }
        map
    }
}

/// High-performance set for natural numbers using bit vector
#[derive(Debug, Clone)]
pub struct NaturalNumberSet {
    bits: Vec<u64>,
    size: usize,
    max_value: usize,
}

impl NaturalNumberSet {
    const BITS_PER_WORD: usize = 64;

    /// Create a new natural number set
    pub fn new() -> Self {
        Self {
            bits: Vec::new(),
            size: 0,
            max_value: 0,
        }
    }

    /// Create a new natural number set with capacity for values up to max_value
    pub fn with_capacity(max_value: usize) -> Self {
        let num_words = (max_value + Self::BITS_PER_WORD) / Self::BITS_PER_WORD;
        Self {
            bits: vec![0; num_words],
            size: 0,
            max_value,
        }
    }

    /// Insert a value into the set
    pub fn insert(&mut self, value: usize) -> bool {
        self.ensure_capacity(value);
        
        let word_index = value / Self::BITS_PER_WORD;
        let bit_index = value % Self::BITS_PER_WORD;
        let mask = 1u64 << bit_index;
        
        let was_present = self.bits[word_index] & mask != 0;
        if !was_present {
            self.bits[word_index] |= mask;
            self.size += 1;
            self.max_value = self.max_value.max(value);
        }
        
        !was_present
    }

    /// Remove a value from the set
    pub fn remove(&mut self, value: usize) -> bool {
        if value > self.max_value || self.bits.is_empty() {
            return false;
        }
        
        let word_index = value / Self::BITS_PER_WORD;
        if word_index >= self.bits.len() {
            return false;
        }
        
        let bit_index = value % Self::BITS_PER_WORD;
        let mask = 1u64 << bit_index;
        
        let was_present = self.bits[word_index] & mask != 0;
        if was_present {
            self.bits[word_index] &= !mask;
            self.size -= 1;
        }
        
        was_present
    }

    /// Check if a value is in the set
    pub fn contains(&self, value: usize) -> bool {
        if value > self.max_value || self.bits.is_empty() {
            return false;
        }
        
        let word_index = value / Self::BITS_PER_WORD;
        if word_index >= self.bits.len() {
            return false;
        }
        
        let bit_index = value % Self::BITS_PER_WORD;
        let mask = 1u64 << bit_index;
        
        self.bits[word_index] & mask != 0
    }

    /// Get the number of elements in the set
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
        self.size = 0;
    }

    /// Iterate over values in the set
    pub fn iter(&self) -> NaturalNumberSetIter {
        NaturalNumberSetIter {
            bits: &self.bits,
            current_word: 0,
            current_bit: 0,
        }
    }

    /// Union with another set
    pub fn union(&mut self, other: &Self) {
        self.ensure_capacity(other.max_value);
        
        for (i, &other_word) in other.bits.iter().enumerate() {
            if i < self.bits.len() {
                self.bits[i] |= other_word;
            }
        }
        
        // Recount size
        self.size = 0;
        for &word in &self.bits {
            self.size += word.count_ones() as usize;
        }
        
        self.max_value = self.max_value.max(other.max_value);
    }

    /// Intersection with another set
    pub fn intersection(&mut self, other: &Self) {
        for (i, word) in self.bits.iter_mut().enumerate() {
            if i < other.bits.len() {
                *word &= other.bits[i];
            } else {
                *word = 0;
            }
        }
        
        // Recount size
        self.size = 0;
        for &word in &self.bits {
            self.size += word.count_ones() as usize;
        }
    }

    /// Difference with another set (self - other)
    pub fn difference(&mut self, other: &Self) {
        for (i, word) in self.bits.iter_mut().enumerate() {
            if i < other.bits.len() {
                *word &= !other.bits[i];
            }
        }
        
        // Recount size
        self.size = 0;
        for &word in &self.bits {
            self.size += word.count_ones() as usize;
        }
    }

    /// Ensure capacity for the given value
    fn ensure_capacity(&mut self, value: usize) {
        let required_words = (value + Self::BITS_PER_WORD) / Self::BITS_PER_WORD;
        if self.bits.len() < required_words {
            self.bits.resize(required_words, 0);
        }
    }

    /// Get the maximum value that can be stored
    pub fn capacity(&self) -> usize {
        self.bits.len() * Self::BITS_PER_WORD
    }
}

impl Default for NaturalNumberSet {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<usize> for NaturalNumberSet {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut set = Self::new();
        for value in iter {
            set.insert(value);
        }
        set
    }
}

/// Iterator for NaturalNumberSet
pub struct NaturalNumberSetIter<'a> {
    bits: &'a [u64],
    current_word: usize,
    current_bit: usize,
}

impl<'a> Iterator for NaturalNumberSetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word < self.bits.len() {
            let word = self.bits[self.current_word];
            
            // Skip to next set bit in current word
            while self.current_bit < NaturalNumberSet::BITS_PER_WORD {
                if word & (1u64 << self.current_bit) != 0 {
                    let value = self.current_word * NaturalNumberSet::BITS_PER_WORD + self.current_bit;
                    self.current_bit += 1;
                    return Some(value);
                }
                self.current_bit += 1;
            }
            
            // Move to next word
            self.current_word += 1;
            self.current_bit = 0;
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_number_map_basic_operations() {
        let mut map = NaturalNumberMap::new();
        
        // Insert values
        assert_eq!(map.insert(0, "zero"), None);
        assert_eq!(map.insert(5, "five"), None);
        assert_eq!(map.insert(10, "ten"), None);
        
        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());
        
        // Get values
        assert_eq!(map.get(0), Some(&"zero"));
        assert_eq!(map.get(5), Some(&"five"));
        assert_eq!(map.get(10), Some(&"ten"));
        assert_eq!(map.get(3), None);
        
        // Update value
        assert_eq!(map.insert(5, "FIVE"), Some("five"));
        assert_eq!(map.len(), 3); // Size shouldn't change
        assert_eq!(map.get(5), Some(&"FIVE"));
        
        // Remove values
        assert_eq!(map.remove(5), Some("FIVE"));
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(5), None);
        
        // Contains check
        assert!(map.contains_key(0));
        assert!(map.contains_key(10));
        assert!(!map.contains_key(5));
    }

    #[test]
    fn test_natural_number_map_indexing() {
        let mut map = NaturalNumberMap::new();
        map.insert(0, 100);
        map.insert(5, 200);
        
        assert_eq!(map[0], 100);
        assert_eq!(map[5], 200);
        
        map[0] = 150;
        assert_eq!(map[0], 150);
    }

    #[test]
    fn test_natural_number_map_iteration() {
        let mut map = NaturalNumberMap::new();
        map.insert(0, "a");
        map.insert(3, "b");
        map.insert(7, "c");
        
        let pairs: Vec<_> = map.iter().collect();
        assert_eq!(pairs, vec![(0, &"a"), (3, &"b"), (7, &"c")]);
        
        let keys: Vec<_> = map.keys().collect();
        assert_eq!(keys, vec![0, 3, 7]);
        
        let values: Vec<_> = map.values().collect();
        assert_eq!(values, vec![&"a", &"b", &"c"]);
    }

    #[test]
    fn test_natural_number_map_from_iterator() {
        let data = vec![(0, "zero"), (2, "two"), (5, "five")];
        let map: NaturalNumberMap<&str> = data.into_iter().collect();
        
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(0), Some(&"zero"));
        assert_eq!(map.get(2), Some(&"two"));
        assert_eq!(map.get(5), Some(&"five"));
    }

    #[test]
    fn test_natural_number_set_basic_operations() {
        let mut set = NaturalNumberSet::new();
        
        // Insert values
        assert!(set.insert(0));
        assert!(set.insert(5));
        assert!(set.insert(10));
        assert!(!set.insert(5)); // Already exists
        
        assert_eq!(set.len(), 3);
        assert!(!set.is_empty());
        
        // Contains check
        assert!(set.contains(0));
        assert!(set.contains(5));
        assert!(set.contains(10));
        assert!(!set.contains(3));
        
        // Remove values
        assert!(set.remove(5));
        assert!(!set.remove(5)); // Already removed
        assert_eq!(set.len(), 2);
        assert!(!set.contains(5));
    }

    #[test]
    fn test_natural_number_set_iteration() {
        let mut set = NaturalNumberSet::new();
        set.insert(0);
        set.insert(3);
        set.insert(7);
        
        let values: Vec<_> = set.iter().collect();
        assert_eq!(values, vec![0, 3, 7]);
    }

    #[test]
    fn test_natural_number_set_from_iterator() {
        let data = vec![0, 2, 5, 7];
        let set: NaturalNumberSet = data.into_iter().collect();
        
        assert_eq!(set.len(), 4);
        assert!(set.contains(0));
        assert!(set.contains(2));
        assert!(set.contains(5));
        assert!(set.contains(7));
        assert!(!set.contains(1));
    }

    #[test]
    fn test_natural_number_set_operations() {
        let mut set1 = NaturalNumberSet::new();
        set1.insert(1);
        set1.insert(2);
        set1.insert(3);
        
        let mut set2 = NaturalNumberSet::new();
        set2.insert(2);
        set2.insert(3);
        set2.insert(4);
        
        // Union
        let mut union_set = set1.clone();
        union_set.union(&set2);
        assert_eq!(union_set.len(), 4);
        assert!(union_set.contains(1));
        assert!(union_set.contains(2));
        assert!(union_set.contains(3));
        assert!(union_set.contains(4));
        
        // Intersection
        let mut intersection_set = set1.clone();
        intersection_set.intersection(&set2);
        assert_eq!(intersection_set.len(), 2);
        assert!(!intersection_set.contains(1));
        assert!(intersection_set.contains(2));
        assert!(intersection_set.contains(3));
        assert!(!intersection_set.contains(4));
        
        // Difference
        let mut difference_set = set1.clone();
        difference_set.difference(&set2);
        assert_eq!(difference_set.len(), 1);
        assert!(difference_set.contains(1));
        assert!(!difference_set.contains(2));
        assert!(!difference_set.contains(3));
        assert!(!difference_set.contains(4));
    }

    #[test]
    fn test_natural_number_set_large_values() {
        let mut set = NaturalNumberSet::new();
        
        // Test with large values
        set.insert(1000);
        set.insert(10000);
        set.insert(100000);
        
        assert_eq!(set.len(), 3);
        assert!(set.contains(1000));
        assert!(set.contains(10000));
        assert!(set.contains(100000));
        assert!(!set.contains(50000));
    }

    #[test]
    fn test_natural_number_map_capacity_growth() {
        let mut map = NaturalNumberMap::new();
        let initial_capacity = map.capacity();
        
        // Insert values that should trigger capacity growth
        for i in 0..initial_capacity + 10 {
            map.insert(i, i * 2);
        }
        
        assert!(map.capacity() > initial_capacity);
        assert_eq!(map.len(), initial_capacity + 10);
        
        // Verify all values are accessible
        for i in 0..initial_capacity + 10 {
            assert_eq!(map.get(i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_natural_number_map_shrink_to_fit() {
        let mut map = NaturalNumberMap::with_capacity(1000);
        map.insert(0, "zero");
        map.insert(5, "five");
        
        assert!(map.capacity() >= 1000);
        map.shrink_to_fit();
        assert!(map.capacity() < 1000);
        assert_eq!(map.len(), 2);
        
        // Values should still be accessible
        assert_eq!(map.get(0), Some(&"zero"));
        assert_eq!(map.get(5), Some(&"five"));
    }
}