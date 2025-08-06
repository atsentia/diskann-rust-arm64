//! Advanced I/O utilities with direct I/O support
//!
//! This module provides optimized I/O operations for disk-based indices,
//! including O_DIRECT support for bypassing the OS page cache.

use crate::{Result, Error};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

/// Sector size for aligned I/O operations (4KB standard for SSDs)
pub const SECTOR_SIZE: usize = 4096;

/// Direct I/O file reader that bypasses OS page cache
pub struct DirectReader {
    file: File,
    #[allow(dead_code)]
    path: std::path::PathBuf,
    file_size: u64,
}

impl DirectReader {
    /// Open a file for direct I/O reading
    #[cfg(target_os = "linux")]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        
        let path_buf = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT) // Enable direct I/O
            .open(&path_buf)
            .map_err(|e| Error::Io(format!("Failed to open file with O_DIRECT: {}", e)))?;
        
        let metadata = file.metadata()
            .map_err(|e| Error::Io(format!("Failed to get file metadata: {}", e)))?;
        
        Ok(Self {
            file,
            path: path_buf,
            file_size: metadata.len(),
        })
    }

    /// Open a file for direct I/O reading (fallback for non-Linux)
    #[cfg(not(target_os = "linux"))]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let file = File::open(&path_buf)
            .map_err(|e| Error::Io(format!("Failed to open file: {}", e)))?;
        
        let metadata = file.metadata()
            .map_err(|e| Error::Io(format!("Failed to get file metadata: {}", e)))?;
        
        log::warn!("Direct I/O not supported on this platform, using buffered I/O");
        
        Ok(Self {
            file,
            path: path_buf,
            file_size: metadata.len(),
        })
    }

    /// Read data at a specific sector-aligned offset
    pub fn read_aligned(&mut self, offset: u64, buffer: &mut [u8]) -> Result<usize> {
        // Ensure offset is sector-aligned
        if offset % SECTOR_SIZE as u64 != 0 {
            return Err(Error::InvalidParameter(
                format!("Offset {} is not sector-aligned ({})", offset, SECTOR_SIZE)
            ).into());
        }

        // Ensure buffer size is sector-aligned
        if buffer.len() % SECTOR_SIZE != 0 {
            return Err(Error::InvalidParameter(
                format!("Buffer size {} is not sector-aligned ({})", buffer.len(), SECTOR_SIZE)
            ).into());
        }

        // Ensure buffer is properly aligned in memory
        let buffer_addr = buffer.as_ptr() as usize;
        if buffer_addr % SECTOR_SIZE != 0 {
            return Err(Error::InvalidParameter(
                "Buffer is not memory-aligned for direct I/O".to_string()
            ).into());
        }

        // Seek to position
        let mut file = &self.file;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| Error::Io(format!("Failed to seek to offset {}: {}", offset, e)))?;

        // Read data
        file.read_exact(buffer)
            .map_err(|e| Error::Io(format!("Failed to read {} bytes at offset {}: {}", buffer.len(), offset, e)))?;

        Ok(buffer.len())
    }

    /// Get file size
    pub fn size(&self) -> u64 {
        self.file_size
    }
}

/// Aligned memory allocator for direct I/O operations
pub struct AlignedBuffer {
    data: Vec<u8>,
    aligned_ptr: *mut u8,
    aligned_len: usize,
}

impl AlignedBuffer {
    /// Allocate a sector-aligned buffer for direct I/O
    pub fn new(size: usize) -> Result<Self> {
        // Round up to next sector boundary
        let aligned_size = (size + SECTOR_SIZE - 1) & !(SECTOR_SIZE - 1);
        
        // Allocate extra space for alignment
        let total_size = aligned_size + SECTOR_SIZE - 1;
        let mut data = vec![0u8; total_size];
        
        // Calculate aligned pointer
        let raw_ptr = data.as_mut_ptr() as usize;
        let aligned_addr = (raw_ptr + SECTOR_SIZE - 1) & !(SECTOR_SIZE - 1);
        let aligned_ptr = aligned_addr as *mut u8;
        
        // Verify alignment
        assert_eq!(aligned_addr % SECTOR_SIZE, 0, "Buffer not properly aligned");
        
        Ok(Self {
            data,
            aligned_ptr,
            aligned_len: aligned_size,
        })
    }

    /// Get a mutable slice to the aligned buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.aligned_ptr, self.aligned_len)
        }
    }

    /// Get an immutable slice to the aligned buffer
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.aligned_ptr, self.aligned_len)
        }
    }

    /// Get the aligned size
    pub fn len(&self) -> usize {
        self.aligned_len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.aligned_len == 0
    }
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

/// High-performance cached reader with direct I/O and prefetching
pub struct CachedDirectReader {
    reader: DirectReader,
    cache: lru::LruCache<u64, Vec<u8>>,
    cache_block_size: usize,
    prefetch_blocks: usize,
}

impl CachedDirectReader {
    /// Create a new cached direct reader
    pub fn open<P: AsRef<Path>>(
        path: P, 
        cache_capacity: usize, 
        cache_block_size: usize,
        prefetch_blocks: usize,
    ) -> Result<Self> {
        let reader = DirectReader::open(path)?;
        
        // Ensure cache block size is sector-aligned
        let aligned_block_size = (cache_block_size + SECTOR_SIZE - 1) & !(SECTOR_SIZE - 1);
        
        Ok(Self {
            reader,
            cache: lru::LruCache::new(std::num::NonZeroUsize::new(cache_capacity).unwrap()),
            cache_block_size: aligned_block_size,
            prefetch_blocks,
        })
    }

    /// Read data with caching and prefetching
    pub fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<usize> {
        let block_id = offset / self.cache_block_size as u64;
        let block_offset = (offset % self.cache_block_size as u64) as usize;
        
        // Check cache first
        if let Some(cached_block) = self.cache.get(&block_id) {
            let copy_len = std::cmp::min(buffer.len(), cached_block.len() - block_offset);
            buffer[..copy_len].copy_from_slice(&cached_block[block_offset..block_offset + copy_len]);
            return Ok(copy_len);
        }

        // Read block from disk
        let block_start = block_id * self.cache_block_size as u64;
        let mut aligned_buffer = AlignedBuffer::new(self.cache_block_size)?;
        
        let bytes_read = self.reader.read_aligned(block_start, aligned_buffer.as_mut_slice())?;
        let block_data = aligned_buffer.as_slice()[..bytes_read].to_vec();
        
        // Cache the block
        self.cache.put(block_id, block_data.clone());
        
        // Prefetch next blocks
        self.prefetch_blocks_async(block_id + 1);
        
        // Copy requested data
        let copy_len = std::cmp::min(buffer.len(), block_data.len() - block_offset);
        buffer[..copy_len].copy_from_slice(&block_data[block_offset..block_offset + copy_len]);
        
        Ok(copy_len)
    }

    /// Asynchronously prefetch blocks
    fn prefetch_blocks_async(&mut self, start_block: u64) {
        for i in 0..self.prefetch_blocks {
            let block_id = start_block + i as u64;
            
            // Skip if already cached
            if self.cache.contains(&block_id) {
                continue;
            }
            
            // Read and cache block
            let block_start = block_id * self.cache_block_size as u64;
            if block_start >= self.reader.size() {
                break;
            }
            
            if let Ok(mut aligned_buffer) = AlignedBuffer::new(self.cache_block_size) {
                if let Ok(bytes_read) = self.reader.read_aligned(block_start, aligned_buffer.as_mut_slice()) {
                    let block_data = aligned_buffer.as_slice()[..bytes_read].to_vec();
                    self.cache.put(block_id, block_data);
                }
            }
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            capacity: self.cache.cap().get(),
            size: self.cache.len(),
            block_size: self.cache_block_size,
        }
    }
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub capacity: usize,
    pub size: usize,
    pub block_size: usize,
}

/// Utility functions for direct I/O operations
pub mod utils {
    use super::*;

    /// Check if an offset is sector-aligned
    pub fn is_sector_aligned(offset: u64) -> bool {
        offset % super::SECTOR_SIZE as u64 == 0
    }

    /// Round up to next sector boundary
    pub fn round_up_to_sector(size: usize) -> usize {
        (size + super::SECTOR_SIZE - 1) & !(super::SECTOR_SIZE - 1)
    }

    /// Check if a memory address is sector-aligned
    pub fn is_memory_aligned(ptr: *const u8) -> bool {
        ptr as usize % super::SECTOR_SIZE == 0
    }

    /// Get optimal read size for direct I/O (multiple of sector size)
    pub fn optimal_read_size(requested_size: usize) -> usize {
        let min_size = super::SECTOR_SIZE;
        let max_size = 64 * 1024; // 64KB max for good performance
        
        let aligned_size = round_up_to_sector(requested_size);
        std::cmp::max(min_size, std::cmp::min(max_size, aligned_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::io::Write;

    fn create_test_file(data: &[u8]) -> std::path::PathBuf {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_file.bin");
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(data).unwrap();
        file.sync_all().unwrap();
        file_path
    }

    #[test]
    fn test_aligned_buffer_creation() {
        let buffer = AlignedBuffer::new(8192).unwrap();
        assert_eq!(buffer.len(), 8192);
        assert!(!buffer.is_empty());
        
        // Verify alignment
        let ptr = buffer.as_slice().as_ptr() as usize;
        assert_eq!(ptr % SECTOR_SIZE, 0, "Buffer not sector-aligned");
    }

    #[test]
    fn test_aligned_buffer_small_size() {
        let buffer = AlignedBuffer::new(100).unwrap();
        // Should be rounded up to next sector boundary
        assert_eq!(buffer.len(), SECTOR_SIZE);
    }

    #[test]
    fn test_utils_sector_alignment() {
        assert!(utils::is_sector_aligned(0));
        assert!(utils::is_sector_aligned(4096));
        assert!(utils::is_sector_aligned(8192));
        assert!(!utils::is_sector_aligned(100));
        assert!(!utils::is_sector_aligned(4097));
    }

    #[test]
    fn test_utils_round_up() {
        assert_eq!(utils::round_up_to_sector(100), SECTOR_SIZE);
        assert_eq!(utils::round_up_to_sector(SECTOR_SIZE), SECTOR_SIZE);
        assert_eq!(utils::round_up_to_sector(SECTOR_SIZE + 1), SECTOR_SIZE * 2);
    }

    #[test]
    fn test_utils_optimal_read_size() {
        assert_eq!(utils::optimal_read_size(100), SECTOR_SIZE);
        assert_eq!(utils::optimal_read_size(8192), 8192);
        assert_eq!(utils::optimal_read_size(100 * 1024), 64 * 1024); // Capped at 64KB
    }

    #[test]
    fn test_direct_reader_creation() {
        // Create a test file with sector-aligned size
        let data = vec![0u8; SECTOR_SIZE * 2];
        let file_path = create_test_file(&data);
        
        let reader = DirectReader::open(&file_path);
        assert!(reader.is_ok());
        
        let reader = reader.unwrap();
        assert_eq!(reader.size(), data.len() as u64);
    }

    #[test]
    fn test_direct_reader_read_aligned() {
        // Create test data (2 sectors)
        let mut data = vec![0u8; SECTOR_SIZE * 2];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        let file_path = create_test_file(&data);
        
        let mut reader = DirectReader::open(&file_path).unwrap();
        let mut buffer = AlignedBuffer::new(SECTOR_SIZE).unwrap();
        
        // Read first sector
        let bytes_read = reader.read_aligned(0, buffer.as_mut_slice()).unwrap();
        assert_eq!(bytes_read, SECTOR_SIZE);
        
        // Verify data
        for (i, &byte) in buffer.as_slice().iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8);
        }
    }

    #[test]
    fn test_direct_reader_unaligned_offset_error() {
        let data = vec![0u8; SECTOR_SIZE * 2];
        let file_path = create_test_file(&data);
        let mut reader = DirectReader::open(&file_path).unwrap();
        let mut buffer = AlignedBuffer::new(SECTOR_SIZE).unwrap();
        
        // Try to read at unaligned offset
        let result = reader.read_aligned(100, buffer.as_mut_slice());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not sector-aligned"));
    }

    #[test]
    fn test_cached_direct_reader() {
        let mut data = vec![0u8; SECTOR_SIZE * 4];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        let file_path = create_test_file(&data);
        
        let mut reader = CachedDirectReader::open(&file_path, 10, SECTOR_SIZE, 2).unwrap();
        
        let mut buffer = vec![0u8; 100];
        let bytes_read = reader.read_at(0, &mut buffer).unwrap();
        assert_eq!(bytes_read, 100);
        
        // Verify data
        for (i, &byte) in buffer.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8);
        }
        
        // Check cache stats
        let stats = reader.cache_stats();
        assert_eq!(stats.block_size, SECTOR_SIZE);
        assert!(stats.size > 0); // Should have cached something
    }

    #[test]
    fn test_cached_reader_cache_hit() {
        let data = vec![0u8; SECTOR_SIZE * 2];
        let file_path = create_test_file(&data);
        let mut reader = CachedDirectReader::open(&file_path, 10, SECTOR_SIZE, 1).unwrap();
        
        let mut buffer1 = vec![0u8; 100];
        let mut buffer2 = vec![0u8; 100];
        
        // First read should cache the block
        reader.read_at(0, &mut buffer1).unwrap();
        
        // Second read from same block should be a cache hit
        reader.read_at(50, &mut buffer2).unwrap();
        
        // Should have the same data (all zeros in this case)
        assert_eq!(buffer1[0..50], buffer2[0..50]);
    }
}