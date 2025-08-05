//! Asynchronous file reader with caching and prefetching
//!
//! This module provides efficient async I/O for disk-based indices.

use crate::{Result, Error};
use memmap2::{Mmap, MmapOptions};
use parking_lot::RwLock;
use hashbrown::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use tokio::fs::File as AsyncFile;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

/// Memory-mapped file reader for efficient random access
pub struct MmapReader {
    mmap: Mmap,
    len: usize,
}

impl MmapReader {
    /// Create a new memory-mapped reader
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let len = metadata.len() as usize;
        
        unsafe {
            let mmap = MmapOptions::new()
                .len(len)
                .map(&file)?;
            
            Ok(Self { mmap, len })
        }
    }
    
    /// Read data at a specific offset
    #[inline]
    pub fn read_at(&self, offset: usize, buf: &mut [u8]) -> Result<()> {
        if offset + buf.len() > self.len {
            return Err(Error::Io("Read beyond end of file".to_string()).into());
        }
        
        buf.copy_from_slice(&self.mmap[offset..offset + buf.len()]);
        Ok(())
    }
    
    /// Get a slice of the memory-mapped data
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Result<&[u8]> {
        if offset + len > self.len {
            return Err(Error::Io("Slice beyond end of file".to_string()).into());
        }
        
        Ok(&self.mmap[offset..offset + len])
    }
    
    /// Get the total file size
    pub fn len(&self) -> usize {
        self.len
    }
}

/// Async file reader with caching
pub struct AsyncReader {
    file: AsyncFile,
    cache: Arc<RwLock<LruCache>>,
    block_size: usize,
}

impl AsyncReader {
    /// Create a new async reader with caching
    pub async fn new<P: AsRef<Path>>(path: P, cache_size: usize, block_size: usize) -> Result<Self> {
        let file = AsyncFile::open(path).await?;
        let cache = Arc::new(RwLock::new(LruCache::new(cache_size)));
        
        Ok(Self {
            file,
            cache,
            block_size,
        })
    }
    
    /// Read data at a specific offset
    pub async fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        // Check cache first
        let block_id = (offset / self.block_size as u64) as usize;
        
        {
            let cache = self.cache.read();
            if let Some(block) = cache.get(block_id) {
                let block_offset = (offset % self.block_size as u64) as usize;
                let copy_len = buf.len().min(block.len() - block_offset);
                buf[..copy_len].copy_from_slice(&block[block_offset..block_offset + copy_len]);
                
                if copy_len == buf.len() {
                    return Ok(());
                }
                // Need to read more data
            }
        }
        
        // Cache miss - read from file
        self.file.seek(std::io::SeekFrom::Start(offset)).await?;
        self.file.read_exact(buf).await?;
        
        // Update cache
        self.cache_block(block_id, offset).await?;
        
        Ok(())
    }
    
    /// Prefetch data into cache
    pub async fn prefetch(&mut self, offset: u64, len: usize) -> Result<()> {
        let start_block = (offset / self.block_size as u64) as usize;
        let end_block = ((offset + len as u64) / self.block_size as u64) as usize;
        
        for block_id in start_block..=end_block {
            if !self.cache.read().contains(block_id) {
                let block_offset = block_id as u64 * self.block_size as u64;
                self.cache_block(block_id, block_offset).await?;
            }
        }
        
        Ok(())
    }
    
    /// Cache a block of data
    async fn cache_block(&mut self, block_id: usize, offset: u64) -> Result<()> {
        let mut block = vec![0u8; self.block_size];
        self.file.seek(std::io::SeekFrom::Start(offset)).await?;
        
        // Read as much as possible
        let bytes_read = match self.file.read(&mut block).await {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(()); // End of file
            }
            Err(e) => return Err(e.into()),
        };
        
        block.truncate(bytes_read);
        self.cache.write().put(block_id, block);
        
        Ok(())
    }
}

/// Simple LRU cache implementation
struct LruCache {
    capacity: usize,
    map: HashMap<usize, (Vec<u8>, usize)>, // block_id -> (data, access_count)
    access_counter: usize,
}

impl LruCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            access_counter: 0,
        }
    }
    
    fn get(&self, key: usize) -> Option<&Vec<u8>> {
        self.map.get(&key).map(|(data, _)| data)
    }
    
    fn contains(&self, key: usize) -> bool {
        self.map.contains_key(&key)
    }
    
    fn put(&mut self, key: usize, value: Vec<u8>) {
        self.access_counter += 1;
        
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            // Evict least recently used
            let lru_key = self.map
                .iter()
                .min_by_key(|(_, (_, access))| access)
                .map(|(k, _)| *k)
                .unwrap();
            self.map.remove(&lru_key);
        }
        
        self.map.insert(key, (value, self.access_counter));
    }
}

/// Buffered reader for sequential access
pub struct BufferedReader {
    reader: MmapReader,
    buffer: Vec<u8>,
    buffer_pos: usize,
    file_pos: usize,
}

impl BufferedReader {
    /// Create a new buffered reader
    pub fn new(reader: MmapReader, buffer_size: usize) -> Self {
        Self {
            reader,
            buffer: vec![0u8; buffer_size],
            buffer_pos: 0,
            file_pos: 0,
        }
    }
    
    /// Read exactly `buf.len()` bytes
    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        let mut bytes_read = 0;
        
        while bytes_read < buf.len() {
            if self.buffer_pos >= self.buffer.len() {
                self.refill_buffer()?;
            }
            
            let available = self.buffer.len() - self.buffer_pos;
            let to_copy = available.min(buf.len() - bytes_read);
            
            buf[bytes_read..bytes_read + to_copy]
                .copy_from_slice(&self.buffer[self.buffer_pos..self.buffer_pos + to_copy]);
            
            self.buffer_pos += to_copy;
            bytes_read += to_copy;
        }
        
        Ok(())
    }
    
    fn refill_buffer(&mut self) -> Result<()> {
        let to_read = self.buffer.len().min(self.reader.len() - self.file_pos);
        if to_read == 0 {
            return Err(Error::Io("End of file".to_string()).into());
        }
        
        self.reader.read_at(self.file_pos, &mut self.buffer[..to_read])?;
        self.file_pos += to_read;
        self.buffer_pos = 0;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[test]
    fn test_mmap_reader() {
        let mut file = NamedTempFile::new().unwrap();
        let data = b"Hello, DiskANN!";
        file.write_all(data).unwrap();
        file.flush().unwrap();
        
        let reader = MmapReader::new(file.path()).unwrap();
        assert_eq!(reader.len(), data.len());
        
        let mut buf = vec![0u8; 5];
        reader.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, b"Hello");
        
        let slice = reader.slice(7, 7).unwrap();
        assert_eq!(slice, b"DiskANN");
    }
    
    #[tokio::test]
    async fn test_async_reader() {
        let mut file = NamedTempFile::new().unwrap();
        let data = b"Hello, DiskANN! This is a test file for async reading.";
        file.write_all(data).unwrap();
        file.flush().unwrap();
        
        let mut reader = AsyncReader::new(file.path(), 2, 16).await.unwrap();
        
        let mut buf = vec![0u8; 7];
        reader.read_at(0, &mut buf).await.unwrap();
        assert_eq!(&buf, b"Hello, ");
        
        // Test cache hit
        let mut buf2 = vec![0u8; 5];
        reader.read_at(0, &mut buf2).await.unwrap();
        assert_eq!(&buf2, b"Hello");
    }
}