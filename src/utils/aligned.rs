//! Aligned memory allocation utilities
//!
//! This module provides utilities for allocating memory with specific alignment
//! requirements, which is important for SIMD operations.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::slice;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;

/// Alignment requirements for different SIMD instruction sets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    /// Default alignment (no special requirements)
    Default,
    /// 16-byte alignment (SSE)
    Align16,
    /// 32-byte alignment (AVX)
    Align32,
    /// 64-byte alignment (cache line)
    Align64,
    /// Custom alignment
    Custom(usize),
}

impl Alignment {
    /// Get the alignment value in bytes
    pub fn bytes(&self) -> usize {
        match self {
            Alignment::Default => std::mem::align_of::<f32>(),
            Alignment::Align16 => 16,
            Alignment::Align32 => 32,
            Alignment::Align64 => 64,
            Alignment::Custom(n) => *n,
        }
    }
    
    /// Get the recommended alignment for the current platform
    pub fn recommended() -> Self {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            Alignment::Align32
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            Alignment::Align16
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            Alignment::Default
        }
    }
}

/// An aligned vector that guarantees memory alignment
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    alignment: usize,
    _marker: PhantomData<T>,
}

impl<T> AlignedVec<T> {
    /// Create a new empty aligned vector
    pub fn new(alignment: Alignment) -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            alignment: alignment.bytes(),
            _marker: PhantomData,
        }
    }
    
    /// Create an aligned vector with capacity
    pub fn with_capacity(capacity: usize, alignment: Alignment) -> Self {
        if capacity == 0 {
            return Self::new(alignment);
        }
        
        let alignment = alignment.bytes();
        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<T>(),
            alignment,
        ).expect("Invalid layout");
        
        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            NonNull::new_unchecked(raw_ptr as *mut T)
        };
        
        Self {
            ptr,
            len: 0,
            capacity,
            alignment,
            _marker: PhantomData,
        }
    }
    
    /// Create an aligned vector filled with a value
    pub fn from_elem(elem: T, count: usize, alignment: Alignment) -> Self
    where
        T: Clone,
    {
        let mut vec = Self::with_capacity(count, alignment);
        vec.resize(count, elem);
        vec
    }
    
    /// Push an element to the vector
    pub fn push(&mut self, value: T) {
        if self.len == self.capacity {
            self.grow();
        }
        
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
            self.len += 1;
        }
    }
    
    /// Resize the vector
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len > self.len {
            self.reserve(new_len - self.len);
            for _ in self.len..new_len {
                self.push(value.clone());
            }
        } else {
            self.truncate(new_len);
        }
    }
    
    /// Truncate the vector
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            unsafe {
                // Drop truncated elements
                for i in len..self.len {
                    self.ptr.as_ptr().add(i).drop_in_place();
                }
            }
            self.len = len;
        }
    }
    
    /// Reserve capacity
    pub fn reserve(&mut self, additional: usize) {
        let required = self.len + additional;
        if required > self.capacity {
            self.grow_to(required);
        }
    }
    
    /// Clear the vector
    pub fn clear(&mut self) {
        self.truncate(0);
    }
    
    /// Get the length
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get the capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get the alignment
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    
    /// Get a pointer to the data
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
    
    /// Get a mutable pointer to the data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
    
    /// Convert to a regular Vec (may involve copying)
    pub fn to_vec(self) -> Vec<T> {
        let mut vec = Vec::with_capacity(self.len);
        unsafe {
            vec.set_len(self.len);
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), vec.as_mut_ptr(), self.len);
        }
        // Prevent double-drop
        std::mem::forget(self);
        vec
    }
    
    /// Grow the vector capacity
    fn grow(&mut self) {
        let new_capacity = if self.capacity == 0 {
            4
        } else {
            self.capacity * 2
        };
        self.grow_to(new_capacity);
    }
    
    /// Grow to a specific capacity
    fn grow_to(&mut self, new_capacity: usize) {
        let new_layout = Layout::from_size_align(
            new_capacity * std::mem::size_of::<T>(),
            self.alignment,
        ).expect("Invalid layout");
        
        let new_ptr = unsafe {
            let raw_ptr = alloc(new_layout);
            if raw_ptr.is_null() {
                std::alloc::handle_alloc_error(new_layout);
            }
            let ptr = NonNull::new_unchecked(raw_ptr as *mut T);
            
            // Copy existing data
            if self.capacity > 0 {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), ptr.as_ptr(), self.len);
                
                // Deallocate old memory
                let old_layout = Layout::from_size_align(
                    self.capacity * std::mem::size_of::<T>(),
                    self.alignment,
                ).unwrap();
                dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
            }
            
            ptr
        };
        
        self.ptr = new_ptr;
        self.capacity = new_capacity;
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                // Drop elements
                for i in 0..self.len {
                    self.ptr.as_ptr().add(i).drop_in_place();
                }
                
                // Deallocate memory
                let layout = Layout::from_size_align(
                    self.capacity * std::mem::size_of::<T>(),
                    self.alignment,
                ).unwrap();
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];
    
    fn deref(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
}

impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
}

// Safe because AlignedVec owns the data
unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

/// Aligned buffer for temporary computations
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    size: usize,
    alignment: usize,
}

impl AlignedBuffer {
    /// Create a new aligned buffer
    pub fn new(size: usize, alignment: Alignment) -> Self {
        let alignment = alignment.bytes();
        let layout = Layout::from_size_align(size, alignment)
            .expect("Invalid layout");
        
        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            NonNull::new_unchecked(raw_ptr)
        };
        
        Self {
            ptr,
            size,
            alignment,
        }
    }
    
    /// Get the size
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get the alignment
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    
    /// Get a slice of the buffer
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(self.ptr.as_ptr(), self.size)
        }
    }
    
    /// Get a mutable slice of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size)
        }
    }
    
    /// Cast to a typed slice
    pub fn as_slice_of<T>(&self) -> &[T] {
        assert!(self.size >= std::mem::size_of::<T>());
        assert!(self.alignment >= std::mem::align_of::<T>());
        
        let count = self.size / std::mem::size_of::<T>();
        unsafe {
            slice::from_raw_parts(self.ptr.as_ptr() as *const T, count)
        }
    }
    
    /// Cast to a mutable typed slice
    pub fn as_mut_slice_of<T>(&mut self) -> &mut [T] {
        assert!(self.size >= std::mem::size_of::<T>());
        assert!(self.alignment >= std::mem::align_of::<T>());
        
        let count = self.size / std::mem::size_of::<T>();
        unsafe {
            slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count)
        }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(self.size, self.alignment).unwrap();
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

/// Check if a pointer is aligned
#[inline]
pub fn is_aligned<T>(ptr: *const T, alignment: Alignment) -> bool {
    ptr as usize % alignment.bytes() == 0
}

/// Align a value up to the nearest multiple of alignment
#[inline]
pub fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aligned_vec() {
        let mut vec: AlignedVec<f32> = AlignedVec::with_capacity(100, Alignment::Align32);
        
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 100);
        assert_eq!(vec.alignment(), 32);
        assert!(is_aligned(vec.as_ptr(), Alignment::Align32));
        
        // Push elements
        for i in 0..50 {
            vec.push(i as f32);
        }
        assert_eq!(vec.len(), 50);
        
        // Access elements
        assert_eq!(vec[0], 0.0);
        assert_eq!(vec[49], 49.0);
        
        // Resize
        vec.resize(100, -1.0);
        assert_eq!(vec.len(), 100);
        assert_eq!(vec[99], -1.0);
    }
    
    #[test]
    fn test_aligned_buffer() {
        let mut buffer = AlignedBuffer::new(1024, Alignment::Align64);
        
        assert_eq!(buffer.size(), 1024);
        assert_eq!(buffer.alignment(), 64);
        
        // Use as f32 slice
        let float_slice = buffer.as_mut_slice_of::<f32>();
        assert_eq!(float_slice.len(), 256); // 1024 / 4
        
        float_slice[0] = 3.14;
        assert_eq!(float_slice[0], 3.14);
    }
    
    #[test]
    fn test_alignment_helpers() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }
}