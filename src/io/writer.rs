//! Asynchronous file writer for index persistence
//!
//! This module provides efficient async I/O for writing disk-based indices.

use crate::{Result, Error};
use std::path::Path;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWriteExt, AsyncSeekExt, BufWriter};
use std::io::SeekFrom;

/// Index file header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IndexHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub num_vectors: u32,
    pub dimension: u32,
    pub max_degree: u32,
    pub metric: u32,
    pub graph_offset: u64,
    pub data_offset: u64,
    pub reserved: [u8; 16],
}

impl IndexHeader {
    pub const MAGIC: &'static [u8; 8] = b"DISKANN\0";
    pub const VERSION: u32 = 1;
    pub const SIZE: usize = std::mem::size_of::<IndexHeader>();
    
    pub fn new(num_vectors: u32, dimension: u32, max_degree: u32, metric: u32) -> Self {
        Self {
            magic: *Self::MAGIC,
            version: Self::VERSION,
            num_vectors,
            dimension,
            max_degree,
            metric,
            graph_offset: Self::SIZE as u64,
            data_offset: 0, // Will be set after graph is written
            reserved: [0; 16],
        }
    }
    
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        unsafe { std::mem::transmute(*self) }
    }
    
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Self::SIZE {
            return Err(Error::Io("Invalid header size".to_string()).into());
        }
        
        let header: Self = unsafe { std::ptr::read(bytes.as_ptr() as *const Self) };
        
        if &header.magic != Self::MAGIC {
            return Err(Error::Io("Invalid magic number".to_string()).into());
        }
        
        if header.version != Self::VERSION {
            return Err(Error::Io(format!("Unsupported version: {}", header.version)).into());
        }
        
        Ok(header)
    }
}

/// Async index writer
pub struct IndexWriter {
    writer: BufWriter<File>,
    header: IndexHeader,
    current_offset: u64,
}

impl IndexWriter {
    /// Create a new index writer
    pub async fn new<P: AsRef<Path>>(
        path: P,
        num_vectors: u32,
        dimension: u32,
        max_degree: u32,
        metric: u32,
    ) -> Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .await?;
        
        let writer = BufWriter::new(file);
        let header = IndexHeader::new(num_vectors, dimension, max_degree, metric);
        
        let mut writer = Self {
            writer,
            header,
            current_offset: 0,
        };
        
        // Write header placeholder
        writer.write_header().await?;
        
        Ok(writer)
    }
    
    /// Write the header
    async fn write_header(&mut self) -> Result<()> {
        self.writer.seek(SeekFrom::Start(0)).await?;
        self.writer.write_all(&self.header.to_bytes()).await?;
        self.current_offset = IndexHeader::SIZE as u64;
        Ok(())
    }
    
    /// Write graph adjacency lists
    pub async fn write_graph(&mut self, graph: &[Vec<usize>]) -> Result<()> {
        self.header.graph_offset = self.current_offset;
        
        for neighbors in graph {
            // Write degree
            let degree = neighbors.len() as u32;
            self.writer.write_all(&degree.to_le_bytes()).await?;
            self.current_offset += 4;
            
            // Write neighbor IDs
            for &neighbor_id in neighbors {
                self.writer.write_all(&(neighbor_id as u32).to_le_bytes()).await?;
                self.current_offset += 4;
            }
        }
        
        self.writer.flush().await?;
        Ok(())
    }
    
    /// Write vector data
    pub async fn write_vectors(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        self.header.data_offset = self.current_offset;
        
        for vector in vectors {
            if vector.len() != self.header.dimension as usize {
                return Err(Error::DimensionMismatch {
                    expected: self.header.dimension as usize,
                    actual: vector.len(),
                }.into());
            }
            
            // Write vector data
            for &value in vector {
                self.writer.write_all(&value.to_le_bytes()).await?;
                self.current_offset += 4;
            }
        }
        
        self.writer.flush().await?;
        Ok(())
    }
    
    /// Finalize the index file
    pub async fn finalize(mut self) -> Result<()> {
        // Update header with final offsets
        self.write_header().await?;
        self.writer.flush().await?;
        Ok(())
    }
}

/// Streaming writer for large datasets
pub struct StreamingWriter {
    writer: IndexWriter,
    buffer: Vec<Vec<f32>>,
    buffer_size: usize,
    vectors_written: usize,
}

impl StreamingWriter {
    /// Create a new streaming writer
    pub async fn new<P: AsRef<Path>>(
        path: P,
        num_vectors: u32,
        dimension: u32,
        max_degree: u32,
        metric: u32,
        buffer_size: usize,
    ) -> Result<Self> {
        let writer = IndexWriter::new(path, num_vectors, dimension, max_degree, metric).await?;
        
        Ok(Self {
            writer,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
            vectors_written: 0,
        })
    }
    
    /// Add a vector to the stream
    pub async fn add_vector(&mut self, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.writer.header.dimension as usize {
            return Err(Error::DimensionMismatch {
                expected: self.writer.header.dimension as usize,
                actual: vector.len(),
            }.into());
        }
        
        self.buffer.push(vector);
        
        if self.buffer.len() >= self.buffer_size {
            self.flush_buffer().await?;
        }
        
        Ok(())
    }
    
    /// Flush buffered vectors to disk
    async fn flush_buffer(&mut self) -> Result<()> {
        if !self.buffer.is_empty() {
            self.writer.write_vectors(&self.buffer).await?;
            self.vectors_written += self.buffer.len();
            self.buffer.clear();
        }
        Ok(())
    }
    
    /// Finalize the streaming write
    pub async fn finalize(mut self, graph: Vec<Vec<usize>>) -> Result<()> {
        // Flush any remaining vectors
        self.flush_buffer().await?;
        
        // Write graph
        self.writer.write_graph(&graph).await?;
        
        // Finalize
        self.writer.finalize().await?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_header_serialization() {
        let header = IndexHeader::new(1000, 128, 32, 0);
        let bytes = header.to_bytes();
        let decoded = IndexHeader::from_bytes(&bytes).unwrap();
        
        assert_eq!(header.magic, decoded.magic);
        assert_eq!(header.version, decoded.version);
        assert_eq!(header.num_vectors, decoded.num_vectors);
        assert_eq!(header.dimension, decoded.dimension);
    }
    
    #[tokio::test]
    async fn test_index_writer() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = IndexWriter::new(file.path(), 2, 3, 16, 0).await.unwrap();
        
        // Write graph
        let graph = vec![vec![1], vec![0]];
        writer.write_graph(&graph).await.unwrap();
        
        // Write vectors
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        writer.write_vectors(&vectors).await.unwrap();
        
        writer.finalize().await.unwrap();
        
        // Verify file was written
        let metadata = tokio::fs::metadata(file.path()).await.unwrap();
        assert!(metadata.len() > IndexHeader::SIZE as u64);
    }
}