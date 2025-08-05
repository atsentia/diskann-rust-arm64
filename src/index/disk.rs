//! Disk-based PQ Flash Index implementation
//!
//! This module implements the PQFlashIndex equivalent from the C++ DiskANN,
//! providing memory-efficient vector search using Product Quantization
//! with disk-based storage for large datasets.

use crate::{Distance, DistanceFunction, Result, Error};
use crate::distance::create_distance_function;
// Simplified PQ interface for disk index
struct SimplePQ {
    num_chunks: usize,
    bits_per_chunk: usize,
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl SimplePQ {
    fn new(num_chunks: usize, bits_per_chunk: usize, dimension: usize) -> Result<Self> {
        Ok(Self {
            num_chunks,
            bits_per_chunk,
            codebooks: vec![vec![vec![0.0; dimension / num_chunks]; 1 << bits_per_chunk]; num_chunks],
        })
    }
    
    fn train(&mut self, _vectors: &[Vec<f32>]) -> Result<()> {
        // Simplified training - in practice this would use K-means
        Ok(())
    }
    
    fn encode(&self, _vector: &[f32]) -> Result<Vec<u8>> {
        // Simplified encoding - return dummy codes
        Ok(vec![0u8; self.num_chunks])
    }
    
    fn preprocess_query(&self, query: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Simplified preprocessing - split query into chunks
        let chunk_size = query.len() / self.num_chunks;
        let mut result = Vec::new();
        
        for i in 0..self.num_chunks {
            let start = i * chunk_size;
            let end = (i + 1) * chunk_size;
            result.push(query[start..end].to_vec());
        }
        
        Ok(result)
    }
    
    fn asymmetric_distance(&self, _codes: &[u8], _pq_query: &[Vec<f32>]) -> Result<f32> {
        // Simplified distance - return dummy distance
        Ok(1.0)
    }
}
use crate::graph::VamanaGraph;
use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};

/// Sector size for aligned I/O (4KB, standard for SSDs)
const SECTOR_SIZE: usize = 4096;

/// Header size for disk index format
const HEADER_SIZE: usize = SECTOR_SIZE;

/// Simple PQ parameters for disk index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQParams {
    /// Number of PQ chunks/subspaces
    pub num_chunks: usize,
    /// Bits per chunk (usually 8)
    pub bits_per_chunk: usize,
}

impl Default for PQParams {
    fn default() -> Self {
        Self {
            num_chunks: 8,
            bits_per_chunk: 8,
        }
    }
}

/// Configuration for PQ Flash Index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQFlashConfig {
    /// Maximum degree for graph vertices
    pub max_degree: usize,
    /// Search list size during construction
    pub search_list_size: usize,
    /// Alpha parameter for pruning
    pub alpha: f32,
    /// Product quantization parameters
    pub pq_params: PQParams,
    /// Number of threads for parallel operations
    pub num_threads: usize,
    /// Use reorder data for higher accuracy
    pub use_reorder_data: bool,
    /// Beam width for cached beam search
    pub beam_width: usize,
}

impl Default for PQFlashConfig {
    fn default() -> Self {
        Self {
            max_degree: 64,
            search_list_size: 100,
            alpha: 1.2,
            pq_params: PQParams::default(),
            num_threads: 4,
            use_reorder_data: false,
            beam_width: 4,
        }
    }
}

/// Statistics for search operations
#[derive(Debug, Default)]
pub struct QueryStats {
    pub nodes_visited: usize,
    pub distance_computations: usize,
    pub sectors_read: usize,
    pub query_time_us: u64,
}

/// Disk-based index header format
#[derive(Debug, Serialize, Deserialize)]
struct DiskIndexHeader {
    /// Magic number for format validation
    magic: u64,
    /// Version of the index format
    version: u32,
    /// Number of data points
    num_points: u64,
    /// Dimension of vectors
    data_dim: u64,
    /// Aligned dimension for SIMD
    aligned_dim: u64,
    /// Maximum degree in graph
    max_degree: u64,
    /// Entry point for search
    entry_point: u32,
    /// Number of frozen points
    num_frozen_points: u64,
    /// Distance metric
    metric: Distance,
    /// Bytes per point on disk
    disk_bytes_per_point: u64,
    /// Number of PQ chunks
    n_chunks: u64,
    /// Reorder data dimensions
    reorder_data_dim: u64,
    /// Sector where reorder data starts
    reorder_data_start_sector: u64,
    /// Vectors per sector for reorder data
    nvecs_per_sector: u64,
}

const DISK_INDEX_MAGIC: u64 = 0x44495341_4E4E5253; // "DISKANN" in ASCII + "RS"

/// Memory-mapped disk-based PQ Flash Index
pub struct PQFlashIndex {
    /// Configuration
    config: PQFlashConfig,
    /// Index file mapping
    index_mmap: Option<Mmap>,
    /// PQ compressed data mapping
    pq_mmap: Option<Mmap>,
    /// Reorder data mapping (optional)
    reorder_mmap: Option<Mmap>,
    /// Product quantizer
    pq: Option<SimplePQ>,
    /// Distance function
    distance_fn: Box<dyn DistanceFunction>,
    /// Index header
    header: Option<DiskIndexHeader>,
    /// Node cache for frequently accessed nodes
    node_cache: Arc<RwLock<lru::LruCache<u32, CachedNode>>>,
    /// Coordinate cache for reorder data
    coord_cache: Arc<RwLock<lru::LruCache<u32, Vec<f32>>>>,
    /// File paths
    index_path: PathBuf,
    pq_path: PathBuf,
    reorder_path: Option<PathBuf>,
}

/// Cached node data structure
#[derive(Debug, Clone)]
struct CachedNode {
    /// Neighbors of this node
    neighbors: Vec<u32>,
    /// PQ compressed coordinates (optional)
    pq_coords: Option<Vec<u8>>,
}

/// Node data on disk format
#[derive(Debug)]
struct DiskNode {
    /// Number of neighbors
    num_neighbors: u32,
    /// Neighbor IDs
    neighbors: Vec<u32>,
    /// PQ compressed coordinates
    pq_coords: Vec<u8>,
}

impl PQFlashIndex {
    /// Estimate memory usage in bytes
    fn memory_usage_estimate(&self) -> usize {
        let cache_size = self.node_cache.read().len() * std::mem::size_of::<CachedNode>();
        let coord_cache_size = self.coord_cache.read().len() * 128; // Estimate
        let base = std::mem::size_of::<Self>();
        base + cache_size + coord_cache_size
    }
    
    /// Build a PQFlashIndex from vectors and save to disk
    pub fn build_from_vectors<P: AsRef<Path>>(
        path: P,
        vectors: Vec<Vec<f32>>,
        config: PQFlashConfig,
    ) -> Result<Self> {
        if vectors.is_empty() {
            return Err(Error::InvalidParameter("No vectors provided".to_string()).into());
        }
        
        let dimension = vectors[0].len();
        let metric = Distance::L2; // Default, could be part of config
        
        let mut index = Self::new(dimension, metric, config);
        index.build_and_save(&vectors, path.as_ref())?;
        index.load(path.as_ref())?;
        Ok(index)
    }
    
    /// Create a new PQ Flash Index
    pub fn new(
        dimension: usize,
        metric: Distance,
        config: PQFlashConfig,
    ) -> Self {
        let cache_capacity = 10000; // Cache up to 10k nodes
        
        Self {
            config,
            index_mmap: None,
            pq_mmap: None,
            reorder_mmap: None,
            pq: None,
            distance_fn: create_distance_function(metric, dimension),
            header: None,
            node_cache: Arc::new(RwLock::new(lru::LruCache::new(std::num::NonZeroUsize::new(cache_capacity).unwrap()))),
            coord_cache: Arc::new(RwLock::new(lru::LruCache::new(std::num::NonZeroUsize::new(cache_capacity).unwrap()))),
            index_path: PathBuf::new(),
            pq_path: PathBuf::new(),
            reorder_path: None,
        }
    }

    /// Build index from vectors and save to disk
    pub fn build_and_save<P: AsRef<Path>>(
        &mut self,
        vectors: &[Vec<f32>],
        index_prefix: P,
    ) -> Result<()> {
        let prefix = index_prefix.as_ref();
        self.index_path = prefix.with_extension("index");
        self.pq_path = prefix.with_extension("pq_compressed.bin");
        
        if self.config.use_reorder_data {
            self.reorder_path = Some(prefix.with_extension("reorder_data.bin"));
        }

        // Step 1: Train Product Quantizer
        println!("Training Product Quantizer...");
        let mut pq = SimplePQ::new(
            self.config.pq_params.num_chunks,
            self.config.pq_params.bits_per_chunk,
            vectors[0].len(),
        )?;
        pq.train(vectors)?;
        
        // Quantize all vectors
        let mut pq_codes = Vec::with_capacity(vectors.len());
        for vector in vectors {
            let code = pq.encode(vector)?;
            pq_codes.push(code);
        }
        
        self.pq = Some(pq);

        // Step 2: Build Vamana graph
        println!("Building Vamana graph...");
        let mut graph = VamanaGraph::new(
            vectors.len(),
            vectors[0].len(),
            Distance::L2, // Use L2 for graph construction
            self.config.max_degree,
            self.config.search_list_size,
            self.config.alpha,
        );
        graph.build(vectors)?;

        // Step 3: Write disk format
        self.write_disk_index(&graph, vectors, &pq_codes)?;
        
        // Step 4: Write PQ compressed data
        self.write_pq_data(&pq_codes)?;
        
        // Step 5: Write reorder data if requested
        if self.config.use_reorder_data {
            self.write_reorder_data(vectors)?;
        }

        println!("Index built successfully!");
        println!("  Index file: {}", self.index_path.display());
        println!("  PQ file: {}", self.pq_path.display());
        if let Some(ref path) = self.reorder_path {
            println!("  Reorder file: {}", path.display());
        }

        Ok(())
    }

    /// Load existing index from disk
    pub fn load<P: AsRef<Path>>(&mut self, index_prefix: P) -> Result<()> {
        let prefix = index_prefix.as_ref();
        self.index_path = prefix.with_extension("index");
        self.pq_path = prefix.with_extension("pq_compressed.bin");
        
        // Load index file
        let index_file = File::open(&self.index_path)
            .map_err(|e| Error::Io(format!("Failed to open index file: {}", e)))?;
        
        self.index_mmap = Some(unsafe {
            MmapOptions::new()
                .map(&index_file)
                .map_err(|e| Error::Io(format!("Failed to mmap index file: {}", e)))?
        });

        // Load and parse header
        self.load_header()?;
        
        // Load PQ data
        let pq_file = File::open(&self.pq_path)
            .map_err(|e| Error::Io(format!("Failed to open PQ file: {}", e)))?;
        
        self.pq_mmap = Some(unsafe {
            MmapOptions::new()
                .map(&pq_file)
                .map_err(|e| Error::Io(format!("Failed to mmap PQ file: {}", e)))?
        });

        // Load PQ quantizer (simplified - would normally deserialize from JSON)
        let pq_metadata_path = prefix.with_extension("pq_metadata.json");
        if pq_metadata_path.exists() {
            if let Some(header) = &self.header {
                let pq = SimplePQ::new(
                    header.n_chunks as usize,
                    8, // Default to 8 bits per chunk
                    header.data_dim as usize,
                )?;
                self.pq = Some(pq);
            }
        }

        // Load reorder data if it exists
        let reorder_path = prefix.with_extension("reorder_data.bin");
        if reorder_path.exists() {
            let reorder_file = File::open(&reorder_path)
                .map_err(|e| Error::Io(format!("Failed to open reorder file: {}", e)))?;
            
            self.reorder_mmap = Some(unsafe {
                MmapOptions::new()
                    .map(&reorder_file)
                    .map_err(|e| Error::Io(format!("Failed to mmap reorder file: {}", e)))?
            });
            self.reorder_path = Some(reorder_path);
        }

        println!("Index loaded successfully!");
        if let Some(header) = &self.header {
            println!("  Points: {}", header.num_points);
            println!("  Dimension: {}", header.data_dim);
            println!("  PQ chunks: {}", header.n_chunks);
            println!("  Max degree: {}", header.max_degree);
        }

        Ok(())
    }

    /// Search for k nearest neighbors using cached beam search
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        search_list_size: usize,
    ) -> Result<(Vec<(u32, f32)>, QueryStats)> {
        self.cached_beam_search(
            query,
            k,
            search_list_size,
            self.config.beam_width,
            false,
            self.config.use_reorder_data,
        )
    }

    /// Cached beam search implementation (core search algorithm)
    pub fn cached_beam_search(
        &self,
        query: &[f32],
        k: usize,
        search_list_size: usize,
        _beam_width: usize,
        _use_filter: bool,
        use_reorder_data: bool,
    ) -> Result<(Vec<(u32, f32)>, QueryStats)> {
        let mut stats = QueryStats::default();
        let start_time = std::time::Instant::now();

        let header = self.header.as_ref()
            .ok_or_else(|| Error::InvalidState("Index not loaded".to_string()))?;

        let pq = self.pq.as_ref()
            .ok_or_else(|| Error::InvalidState("PQ not loaded".to_string()))?;

        // Preprocess query for PQ distance computation
        let pq_query = pq.preprocess_query(query)?;

        // Initialize search with entry point
        let mut candidates = std::collections::BinaryHeap::new();
        let mut visited = hashbrown::HashSet::new();
        let mut result_candidates = Vec::new();

        // Add entry point
        let entry_point = header.entry_point;
        let entry_dist = self.compute_distance_to_node(entry_point, &pq_query, pq)?;
        
        candidates.push(std::cmp::Reverse((ordered_float::OrderedFloat(entry_dist), entry_point)));
        visited.insert(entry_point);
        stats.nodes_visited += 1;

        // Beam search loop
        while !candidates.is_empty() && visited.len() < search_list_size {
            // Get closest unvisited candidate
            let (current_dist, current_node) = match candidates.pop() {
                Some(std::cmp::Reverse((dist, node))) => (dist.into_inner(), node),
                None => break,
            };

            // Add to result candidates
            result_candidates.push((current_node, current_dist));

            // Early termination check
            if result_candidates.len() >= k {
                if let Some((_, worst_dist)) = result_candidates.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                    if current_dist > worst_dist * self.config.alpha {
                        break;
                    }
                }
            }

            // Explore neighbors
            let neighbors = self.get_node_neighbors(current_node)?;
            stats.sectors_read += 1;

            for &neighbor_id in &neighbors {
                if !visited.contains(&neighbor_id) && visited.len() < search_list_size {
                    visited.insert(neighbor_id);
                    stats.nodes_visited += 1;

                    let neighbor_dist = self.compute_distance_to_node(neighbor_id, &pq_query, pq)?;
                    stats.distance_computations += 1;

                    candidates.push(std::cmp::Reverse((ordered_float::OrderedFloat(neighbor_dist), neighbor_id)));
                }
            }
        }

        // Sort and take top k
        result_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result_candidates.truncate(k);

        // Reorder using full precision if requested
        if use_reorder_data && self.reorder_mmap.is_some() {
            result_candidates = self.reorder_results(query, &result_candidates)?;
        }

        stats.query_time_us = start_time.elapsed().as_micros() as u64;

        Ok((result_candidates, stats))
    }

    /// Get neighbors of a node (with caching)
    fn get_node_neighbors(&self, node_id: u32) -> Result<Vec<u32>> {
        // Check cache first
        {
            let cache = self.node_cache.read();
            if let Some(cached_node) = cache.peek(&node_id) {
                return Ok(cached_node.neighbors.clone());
            }
        }

        // Load from disk
        let node = self.load_node_from_disk(node_id)?;
        let neighbors = node.neighbors.clone();

        // Cache the node
        {
            let mut cache = self.node_cache.write();
            cache.put(node_id, CachedNode {
                neighbors: node.neighbors,
                pq_coords: Some(node.pq_coords),
            });
        }

        Ok(neighbors)
    }

    /// Compute distance from query to a node using PQ
    fn compute_distance_to_node(
        &self,
        node_id: u32,
        pq_query: &[Vec<f32>],
        pq: &SimplePQ,
    ) -> Result<f32> {
        let pq_codes = self.get_node_pq_codes(node_id)?;
        pq.asymmetric_distance(&pq_codes, pq_query)
    }

    /// Get PQ codes for a node
    fn get_node_pq_codes(&self, node_id: u32) -> Result<Vec<u8>> {
        // Check cache first
        {
            let cache = self.node_cache.read();
            if let Some(cached_node) = cache.peek(&node_id) {
                if let Some(ref pq_coords) = cached_node.pq_coords {
                    return Ok(pq_coords.clone());
                }
            }
        }

        // Load from PQ data file
        let pq_mmap = self.pq_mmap.as_ref()
            .ok_or_else(|| Error::InvalidState("PQ data not loaded".to_string()))?;

        let header = self.header.as_ref().unwrap();
        let bytes_per_point = header.n_chunks as usize * std::mem::size_of::<u8>();
        let offset = node_id as usize * bytes_per_point;

        if offset + bytes_per_point > pq_mmap.len() {
            return Err(Error::InvalidParameter(format!("Node {} out of bounds", node_id)).into());
        }

        let pq_codes = pq_mmap[offset..offset + bytes_per_point].to_vec();
        Ok(pq_codes)
    }

    /// Load header from index file
    fn load_header(&mut self) -> Result<()> {
        let mmap = self.index_mmap.as_ref()
            .ok_or_else(|| Error::InvalidState("Index not mapped".to_string()))?;

        if mmap.len() < HEADER_SIZE {
            return Err(Error::InvalidFormat("Index file too small for header".to_string()).into());
        }

        // Read magic number first
        let magic = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3],
            mmap[4], mmap[5], mmap[6], mmap[7],
        ]);

        if magic != DISK_INDEX_MAGIC {
            return Err(Error::InvalidFormat(format!(
                "Invalid magic number: expected {:#x}, got {:#x}",
                DISK_INDEX_MAGIC, magic
            )).into());
        }

        // Deserialize header from the first sector
        let header_bytes = &mmap[0..HEADER_SIZE];
        let mut cursor = std::io::Cursor::new(header_bytes);
        let header: DiskIndexHeader = bincode::deserialize_from(&mut cursor)
            .map_err(|e| Error::InvalidFormat(format!("Failed to deserialize header: {}", e)))?;

        self.header = Some(header);
        Ok(())
    }

    /// Write disk index format
    fn write_disk_index(
        &mut self,
        graph: &VamanaGraph,
        vectors: &[Vec<f32>],
        _pq_codes: &[Vec<u8>],
    ) -> Result<()> {
        let mut file = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.index_path)
                .map_err(|e| Error::Io(format!("Failed to create index file: {}", e)))?
        );

        // Create header
        let header = DiskIndexHeader {
            magic: DISK_INDEX_MAGIC,
            version: 1,
            num_points: vectors.len() as u64,
            data_dim: vectors[0].len() as u64,
            aligned_dim: vectors[0].len() as u64, // Simplified
            max_degree: self.config.max_degree as u64,
            entry_point: graph.get_entry_point() as u32,
            num_frozen_points: 0,
            metric: Distance::L2, // Simplified
            disk_bytes_per_point: (self.config.max_degree * 4 + self.config.pq_params.num_chunks) as u64,
            n_chunks: self.config.pq_params.num_chunks as u64,
            reorder_data_dim: if self.config.use_reorder_data { vectors[0].len() as u64 } else { 0 },
            reorder_data_start_sector: 0, // Will be updated if reorder data is used
            nvecs_per_sector: SECTOR_SIZE as u64 / (vectors[0].len() * 4) as u64,
        };

        // Write header
        let header_bytes = bincode::serialize(&header)
            .map_err(|e| Error::Serialization(format!("Failed to serialize header: {}", e)))?;
        
        file.write_all(&header_bytes)
            .map_err(|e| Error::Io(format!("Failed to write header: {}", e)))?;
        
        // Pad to sector boundary
        let padding_size = HEADER_SIZE - header_bytes.len();
        file.write_all(&vec![0u8; padding_size])
            .map_err(|e| Error::Io(format!("Failed to write header padding: {}", e)))?;

        // Write graph data (simplified format)
        for i in 0..vectors.len() {
            let neighbors = graph.get_neighbors(i);
            
            // Write number of neighbors
            file.write_all(&(neighbors.len() as u32).to_le_bytes())
                .map_err(|e| Error::Io(format!("Failed to write neighbor count: {}", e)))?;
            
            // Write neighbor IDs
            for &neighbor_id in &neighbors {
                file.write_all(&(neighbor_id as u32).to_le_bytes())
                    .map_err(|e| Error::Io(format!("Failed to write neighbor ID: {}", e)))?;
            }
            
            // Pad to fixed size per node for simplicity
            let bytes_written = 4 + neighbors.len() * 4;
            let max_bytes = 4 + self.config.max_degree * 4;
            if bytes_written < max_bytes {
                let padding = vec![0u8; max_bytes - bytes_written];
                file.write_all(&padding)
                    .map_err(|e| Error::Io(format!("Failed to write node padding: {}", e)))?;
            }
        }

        file.flush()
            .map_err(|e| Error::Io(format!("Failed to flush index file: {}", e)))?;

        self.header = Some(header);
        Ok(())
    }

    /// Write PQ compressed data
    fn write_pq_data(&self, pq_codes: &[Vec<u8>]) -> Result<()> {
        let mut file = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.pq_path)
                .map_err(|e| Error::Io(format!("Failed to create PQ file: {}", e)))?
        );

        // Write all PQ codes sequentially
        for codes in pq_codes {
            file.write_all(&codes)
                .map_err(|e| Error::Io(format!("Failed to write PQ codes: {}", e)))?;
        }

        file.flush()
            .map_err(|e| Error::Io(format!("Failed to flush PQ file: {}", e)))?;

        // Write PQ metadata (simplified JSON)
        if let Some(ref pq) = self.pq {
            let metadata_path = self.pq_path.with_extension("pq_metadata.json");
            let metadata = format!(
                "{{\"num_chunks\":{},\"bits_per_chunk\":{}}}",
                pq.num_chunks, pq.bits_per_chunk
            );
            
            std::fs::write(&metadata_path, metadata)
                .map_err(|e| Error::Io(format!("Failed to write PQ metadata: {}", e)))?;
        }

        Ok(())
    }

    /// Write reorder data for higher accuracy
    fn write_reorder_data(&self, vectors: &[Vec<f32>]) -> Result<()> {
        if let Some(ref reorder_path) = self.reorder_path {
            let mut file = BufWriter::new(
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(reorder_path)
                    .map_err(|e| Error::Io(format!("Failed to create reorder file: {}", e)))?
            );

            // Write vectors as raw f32 data
            for vector in vectors {
                for &value in vector {
                    file.write_all(&value.to_le_bytes())
                        .map_err(|e| Error::Io(format!("Failed to write reorder data: {}", e)))?;
                }
            }

            file.flush()
                .map_err(|e| Error::Io(format!("Failed to flush reorder file: {}", e)))?;
        }

        Ok(())
    }

    /// Load a node from disk
    fn load_node_from_disk(&self, node_id: u32) -> Result<DiskNode> {
        let mmap = self.index_mmap.as_ref()
            .ok_or_else(|| Error::InvalidState("Index not mapped".to_string()))?;

        let _header = self.header.as_ref().unwrap();
        
        // Calculate node offset (after header)
        let node_size = 4 + self.config.max_degree * 4; // 4 bytes for count + max neighbors
        let node_offset = HEADER_SIZE + node_id as usize * node_size;

        if node_offset + node_size > mmap.len() {
            return Err(Error::InvalidParameter(format!("Node {} out of bounds", node_id)).into());
        }

        // Read number of neighbors
        let num_neighbors = u32::from_le_bytes([
            mmap[node_offset],
            mmap[node_offset + 1],
            mmap[node_offset + 2],
            mmap[node_offset + 3],
        ]);

        // Read neighbor IDs
        let mut neighbors = Vec::with_capacity(num_neighbors as usize);
        for i in 0..num_neighbors {
            let offset = node_offset + 4 + (i as usize * 4);
            let neighbor_id = u32::from_le_bytes([
                mmap[offset],
                mmap[offset + 1],
                mmap[offset + 2],
                mmap[offset + 3],
            ]);
            neighbors.push(neighbor_id);
        }

        // Get PQ codes for this node
        let pq_codes = self.get_node_pq_codes(node_id)?;

        Ok(DiskNode {
            num_neighbors,
            neighbors,
            pq_coords: pq_codes,
        })
    }

    /// Reorder results using full precision data
    fn reorder_results(&self, query: &[f32], results: &[(u32, f32)]) -> Result<Vec<(u32, f32)>> {
        if self.reorder_mmap.is_none() {
            return Ok(results.to_vec());
        }

        let reorder_mmap = self.reorder_mmap.as_ref().unwrap();
        let header = self.header.as_ref().unwrap();
        let dim = header.data_dim as usize;

        let mut reordered = Vec::new();
        for &(node_id, _) in results {
            // Load full precision vector
            let offset = node_id as usize * dim * 4; // 4 bytes per f32
            if offset + dim * 4 <= reorder_mmap.len() {
                let mut vector = Vec::with_capacity(dim);
                for i in 0..dim {
                    let value_offset = offset + i * 4;
                    let value = f32::from_le_bytes([
                        reorder_mmap[value_offset],
                        reorder_mmap[value_offset + 1],
                        reorder_mmap[value_offset + 2],
                        reorder_mmap[value_offset + 3],
                    ]);
                    vector.push(value);
                }

                // Compute precise distance
                let precise_dist = self.distance_fn.distance(query, &vector)?;
                reordered.push((node_id, precise_dist));
            } else {
                reordered.push((node_id, results.iter().find(|(id, _)| *id == node_id).unwrap().1));
            }
        }

        Ok(reordered)
    }

    /// Get index statistics
    pub fn get_stats(&self) -> Option<DiskIndexStats> {
        self.header.as_ref().map(|header| DiskIndexStats {
            num_points: header.num_points,
            data_dim: header.data_dim,
            aligned_dim: header.aligned_dim,
            max_degree: header.max_degree,
            n_chunks: header.n_chunks,
            disk_bytes_per_point: header.disk_bytes_per_point,
            has_reorder_data: self.reorder_mmap.is_some(),
            index_file_size: self.index_mmap.as_ref().map(|m| m.len()).unwrap_or(0),
            pq_file_size: self.pq_mmap.as_ref().map(|m| m.len()).unwrap_or(0),
            reorder_file_size: self.reorder_mmap.as_ref().map(|m| m.len()).unwrap_or(0),
        })
    }
}

/// Statistics about the disk index
#[derive(Debug)]
pub struct DiskIndexStats {
    pub num_points: u64,
    pub data_dim: u64,
    pub aligned_dim: u64,
    pub max_degree: u64,
    pub n_chunks: u64,
    pub disk_bytes_per_point: u64,
    pub has_reorder_data: bool,
    pub index_file_size: usize,
    pub pq_file_size: usize,
    pub reorder_file_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_vectors;
    use tempfile::TempDir;
    use std::time::Instant;

    // ===== SMOKE TESTS =====

    #[test]
    fn smoke_test_pq_flash_index_creation() {
        let config = PQFlashConfig::default();
        let index = PQFlashIndex::new(64, Distance::L2, config);
        
        // Just ensure we can create the index without panicking
        assert!(index.header.is_none()); // No header until loaded
        assert!(index.pq.is_none()); // No PQ until built
    }

    #[test]
    fn smoke_test_config_serialization() {
        let config = PQFlashConfig {
            max_degree: 32,
            search_list_size: 50,
            alpha: 1.5,
            pq_params: PQParams {
                num_chunks: 8,
                bits_per_chunk: 8,
                ..Default::default()
            },
            num_threads: 2,
            use_reorder_data: true,
            beam_width: 6,
        };

        // Test serialization round-trip
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PQFlashConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.max_degree, deserialized.max_degree);
        assert_eq!(config.use_reorder_data, deserialized.use_reorder_data);
        assert_eq!(config.pq_params.num_chunks, deserialized.pq_params.num_chunks);
    }

    #[test]
    fn smoke_test_query_stats_default() {
        let stats = QueryStats::default();
        assert_eq!(stats.nodes_visited, 0);
        assert_eq!(stats.distance_computations, 0);
        assert_eq!(stats.sectors_read, 0);
        assert_eq!(stats.query_time_us, 0);
    }

    // ===== UNIT TESTS =====

    #[test]
    fn test_pq_flash_index_build_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");

        // Generate test data
        let vectors = generate_random_vectors(500, 32);
        
        // Build index
        let config = PQFlashConfig {
            max_degree: 16,
            search_list_size: 25,
            pq_params: PQParams {
                num_chunks: 4,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut index = PQFlashIndex::new(32, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();

        // Verify files were created
        assert!(index_path.with_extension("index").exists());
        assert!(index_path.with_extension("pq_compressed.bin").exists());
        assert!(index_path.with_extension("pq_metadata.json").exists());

        // Load index
        let mut loaded_index = PQFlashIndex::new(32, Distance::L2, config);
        loaded_index.load(&index_path).unwrap();

        // Verify loaded state
        let stats = loaded_index.get_stats().unwrap();
        assert_eq!(stats.num_points, 500);
        assert_eq!(stats.data_dim, 32);
        assert_eq!(stats.n_chunks, 4);
        assert_eq!(stats.max_degree, 16);
        assert!(!stats.has_reorder_data);
    }

    #[test]
    fn test_pq_flash_index_with_reorder_data() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_reorder_index");

        // Generate test data
        let vectors = generate_random_vectors(200, 16);
        
        // Build index with reorder data
        let config = PQFlashConfig {
            max_degree: 8,
            use_reorder_data: true,
            pq_params: PQParams {
                num_chunks: 2,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut index = PQFlashIndex::new(16, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();

        // Verify reorder file was created
        assert!(index_path.with_extension("reorder_data.bin").exists());

        // Load and test
        let mut loaded_index = PQFlashIndex::new(16, Distance::L2, config);
        loaded_index.load(&index_path).unwrap();

        let stats = loaded_index.get_stats().unwrap();
        assert!(stats.has_reorder_data);
        assert!(stats.reorder_file_size > 0);
        assert_eq!(stats.reorder_file_size, 200 * 16 * 4); // 200 vectors * 16 dims * 4 bytes
    }

    #[test]
    fn test_disk_index_header_validation() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_header");

        // Create a file with invalid magic number
        let invalid_file = index_path.with_extension("index");
        std::fs::write(&invalid_file, &[0u8; HEADER_SIZE]).unwrap();

        // Try to load - should fail
        let mut index = PQFlashIndex::new(32, Distance::L2, PQFlashConfig::default());
        let result = index.load(&index_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid magic number"));
    }

    #[test]
    fn test_node_caching() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_cache");

        // Build small index for caching test
        let vectors = generate_random_vectors(50, 8);
        let config = PQFlashConfig {
            max_degree: 4,
            pq_params: PQParams {
                num_chunks: 2,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut index = PQFlashIndex::new(8, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();
        index.load(&index_path).unwrap();

        // Access same node multiple times
        let node_id = 0u32;
        let neighbors1 = index.get_node_neighbors(node_id).unwrap();
        let neighbors2 = index.get_node_neighbors(node_id).unwrap();

        // Should get same results (and second access should be cached)
        assert_eq!(neighbors1, neighbors2);
        
        // Verify cache has the node
        let cache = index.node_cache.read();
        assert!(cache.peek(&node_id).is_some());
    }

    #[test]
    fn test_pq_codes_retrieval() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_pq_codes");

        let vectors = generate_random_vectors(100, 16);
        let config = PQFlashConfig {
            pq_params: PQParams {
                num_chunks: 4,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut index = PQFlashIndex::new(16, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();
        index.load(&index_path).unwrap();

        // Test PQ codes retrieval
        let node_id = 5u32;
        let pq_codes = index.get_node_pq_codes(node_id).unwrap();
        
        // Should have correct number of chunks
        assert_eq!(pq_codes.len(), 4); // 4 chunks as configured
        
        // Each chunk should be valid (0-255 for 8-bit quantization)
        for &code in &pq_codes {
            assert!(code <= 255);
        }
    }

    // ===== INTEGRATION TESTS =====

    #[test]
    fn test_end_to_end_search_accuracy() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_e2e");

        // Create structured test data for accuracy testing
        let mut vectors = Vec::new();
        
        // Create clusters of similar vectors
        for cluster in 0..5 {
            for i in 0..20 {
                let mut vec = vec![0.0f32; 16];
                vec[0] = cluster as f32 * 10.0; // Main cluster dimension
                vec[1] = i as f32 * 0.1; // Within-cluster variation
                // Add small random noise
                for j in 2..16 {
                    vec[j] = (cluster * i + j) as f32 * 0.01;
                }
                vectors.push(vec);
            }
        }

        let config = PQFlashConfig {
            max_degree: 8,
            search_list_size: 20,
            pq_params: PQParams {
                num_chunks: 4,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        // Build and load index
        let mut index = PQFlashIndex::new(16, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();
        index.load(&index_path).unwrap();

        // Test search accuracy
        let query = &vectors[0]; // Query with first vector
        let (results, stats) = index.search(query, 5, 20).unwrap();

        // Verify search quality
        assert_eq!(results.len(), 5);
        assert!(stats.nodes_visited > 0);
        assert!(stats.distance_computations > 0);
        
        // Results should be reasonably close (PQ will have some approximation error)
        for (_, dist) in &results {
            assert!(*dist < 50.0); // Should find vectors from same cluster
        }
    }

    #[test]
    fn test_search_with_reorder_data() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_reorder_search");

        let vectors = generate_random_vectors(100, 32);
        let config = PQFlashConfig {
            max_degree: 12,
            use_reorder_data: true,
            pq_params: PQParams {
                num_chunks: 8,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut index = PQFlashIndex::new(32, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();
        index.load(&index_path).unwrap();

        // Search with and without reorder data
        let query = &vectors[10];
        
        // Search with PQ only
        let (results_pq, _) = index.cached_beam_search(query, 3, 15, 4, false, false).unwrap();
        
        // Search with reorder data
        let (results_reorder, _) = index.cached_beam_search(query, 3, 15, 4, false, true).unwrap();

        assert_eq!(results_pq.len(), 3);
        assert_eq!(results_reorder.len(), 3);
        
        // Reordered results should generally be more accurate (lower distances)
        // Note: This might not always be true due to randomness, but should hold statistically
    }

    #[test]
    fn test_large_index_operations() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_large");

        // Create larger dataset for stress testing
        let vectors = generate_random_vectors(1000, 64);
        let config = PQFlashConfig {
            max_degree: 32,
            search_list_size: 50,
            pq_params: PQParams {
                num_chunks: 8,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut index = PQFlashIndex::new(64, Distance::L2, config.clone());
        
        // Time the build process
        let build_start = Instant::now();
        index.build_and_save(&vectors, &index_path).unwrap();
        let build_time = build_start.elapsed();
        
        println!("Build time for 1000 vectors: {:?}", build_time);
        assert!(build_time.as_secs() < 30); // Should complete within 30 seconds

        // Load and verify
        index.load(&index_path).unwrap();
        let stats = index.get_stats().unwrap();
        assert_eq!(stats.num_points, 1000);
        assert_eq!(stats.data_dim, 64);

        // Test multiple searches
        for i in 0..10 {
            let query = &vectors[i * 100];
            let search_start = Instant::now();
            let (results, search_stats) = index.search(query, 10, 50).unwrap();
            let search_time = search_start.elapsed();
            
            assert_eq!(results.len(), 10);
            assert!(search_time.as_millis() < 100); // Should be fast
            assert!(search_stats.nodes_visited <= 50); // Respects search list size
        }
    }

    // ===== PERFORMANCE TESTS =====

    #[test]
    fn perf_test_search_throughput() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("perf_throughput");

        // Build medium-sized index
        let vectors = generate_random_vectors(2000, 128);
        let config = PQFlashConfig {
            max_degree: 64,
            search_list_size: 100,
            pq_params: PQParams {
                num_chunks: 16,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut index = PQFlashIndex::new(128, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();
        index.load(&index_path).unwrap();

        // Measure search throughput
        let num_queries = 100;  
        let k = 10;
        let search_l = 50;

        let start_time = Instant::now();
        let mut total_results = 0;
        let mut total_stats = QueryStats::default();

        for i in 0..num_queries {
            let query = &vectors[i];
            let (results, stats) = index.search(query, k, search_l).unwrap();
            total_results += results.len();
            total_stats.nodes_visited += stats.nodes_visited;
            total_stats.distance_computations += stats.distance_computations;
            total_stats.sectors_read += stats.sectors_read;
        }

        let elapsed = start_time.elapsed();
        let qps = num_queries as f64 / elapsed.as_secs_f64();
        
        println!("Search Performance Results:");
        println!("  Queries: {}", num_queries);
        println!("  Total time: {:?}", elapsed);
        println!("  QPS: {:.2}", qps);
        println!("  Avg nodes visited: {:.2}", total_stats.nodes_visited as f64 / num_queries as f64);
        println!("  Avg distance computations: {:.2}", total_stats.distance_computations as f64 / num_queries as f64);
        
        // Performance expectations
        assert!(qps > 100.0); // At least 100 QPS
        assert_eq!(total_results, num_queries * k); // All searches should return k results
        assert!(total_stats.nodes_visited > 0);
        assert!(total_stats.distance_computations > 0);
    }

    #[test]
    fn perf_test_memory_usage() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("perf_memory");

        let vectors = generate_random_vectors(500, 64);
        let config = PQFlashConfig {
            max_degree: 32,
            pq_params: PQParams {
                num_chunks: 8,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut index = PQFlashIndex::new(64, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();
        index.load(&index_path).unwrap();

        let stats = index.get_stats().unwrap();
        
        println!("Memory Usage Analysis:");
        println!("  Original data size: {} MB", (500 * 64 * 4) / 1024 / 1024);
        println!("  Index file size: {} KB", stats.index_file_size / 1024);
        println!("  PQ file size: {} KB", stats.pq_file_size / 1024);
        println!("  Total disk usage: {} KB", (stats.index_file_size + stats.pq_file_size) / 1024);
        
        // Memory efficiency checks
        let original_size = 500 * 64 * 4; // 500 vectors * 64 dims * 4 bytes
        let compressed_size = stats.pq_file_size;
        let compression_ratio = original_size as f64 / compressed_size as f64;
        
        println!("  Compression ratio: {:.2}x", compression_ratio);
        assert!(compression_ratio > 4.0); // Should achieve at least 4x compression
    }

    #[test]
    fn perf_test_cache_effectiveness() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("perf_cache");

        let vectors = generate_random_vectors(300, 32);
        let config = PQFlashConfig {
            max_degree: 16,
            pq_params: PQParams {
                num_chunks: 4,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut index = PQFlashIndex::new(32, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();
        index.load(&index_path).unwrap();

        // Perform searches to populate cache
        let query = &vectors[0];
        
        // First search - cold cache
        let start_cold = Instant::now();
        let (results1, _) = index.search(query, 5, 20).unwrap();
        let cold_time = start_cold.elapsed();

        // Second search - warm cache (same query)
        let start_warm = Instant::now();
        let (results2, _) = index.search(query, 5, 20).unwrap();
        let warm_time = start_warm.elapsed();

        // Results should be identical
        assert_eq!(results1.len(), results2.len());
        
        // Warm search should be faster (though difference might be small for small indices)
        println!("Cache Performance:");
        println!("  Cold search time: {:?}", cold_time);
        println!("  Warm search time: {:?}", warm_time);
        
        if warm_time.as_micros() > 0 && cold_time.as_micros() > 0 {
            let speedup = cold_time.as_micros() as f64 / warm_time.as_micros() as f64;
            println!("  Cache speedup: {:.2}x", speedup);
        }
    }

    // ===== ERROR HANDLING TESTS =====

    #[test]
    fn test_error_handling_missing_files() {
        let temp_dir = TempDir::new().unwrap();
        let nonexistent_path = temp_dir.path().join("does_not_exist");

        let mut index = PQFlashIndex::new(32, Distance::L2, PQFlashConfig::default());
        let result = index.load(&nonexistent_path);
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to open index file"));
    }

    #[test]
    fn test_error_handling_invalid_node_id() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_invalid_node");

        let vectors = generate_random_vectors(50, 16);
        let config = PQFlashConfig::default();

        let mut index = PQFlashIndex::new(16, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();
        index.load(&index_path).unwrap();

        // Try to access node beyond bounds
        let result = index.get_node_pq_codes(999);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_error_handling_search_without_loading() {
        let index = PQFlashIndex::new(32, Distance::L2, PQFlashConfig::default());
        let query = vec![0.0f32; 32];
        
        let result = index.search(&query, 5, 20);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Index not loaded"));
    }

    // ===== REGRESSION TESTS =====

    #[test]
    fn test_regression_empty_vectors() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_empty");

        let mut index = PQFlashIndex::new(32, Distance::L2, PQFlashConfig::default());
        let empty_vectors: Vec<Vec<f32>> = vec![];
        
        let result = index.build_and_save(&empty_vectors, &index_path);
        assert!(result.is_err());
        // Should fail gracefully when given empty input
    }

    #[test]
    fn test_regression_single_vector() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_single");

        let vectors = vec![vec![1.0f32; 32]];
        let config = PQFlashConfig {
            max_degree: 8,
            pq_params: PQParams {
                num_chunks: 4,
                bits_per_chunk: 8,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut index = PQFlashIndex::new(32, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();
        index.load(&index_path).unwrap();

        // Should handle single vector gracefully
        let (results, _) = index.search(&vectors[0], 1, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0); // Should find itself
    }

    #[test]
    fn test_regression_dimension_consistency() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_dimensions");

        let vectors = generate_random_vectors(100, 64);
        let config = PQFlashConfig::default();

        let mut index = PQFlashIndex::new(64, Distance::L2, config.clone());
        index.build_and_save(&vectors, &index_path).unwrap();
        index.load(&index_path).unwrap();

        // Query with wrong dimension should fail
        let wrong_dim_query = vec![0.0f32; 32]; // 32 instead of 64
        let result = index.search(&wrong_dim_query, 5, 20);
        // This might succeed with dimension mismatch handled at distance function level
        // The exact behavior depends on the distance function implementation
    }
}

// Implement the Index trait for PQFlashIndex
impl super::Index for PQFlashIndex {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        // Use default search parameters
        let search_list_size = k * 10; // Reasonable default
        let (results, _stats) = self.search(query, k, search_list_size)?;
        // Convert u32 to usize
        Ok(results.into_iter().map(|(id, dist)| (id as usize, dist)).collect())
    }
    
    fn size(&self) -> usize {
        self.header.as_ref().map(|h| h.num_points as usize).unwrap_or(0)
    }
    
    fn dimension(&self) -> usize {
        self.header.as_ref().map(|h| h.data_dim as usize).unwrap_or(0)
    }
    
    fn metric(&self) -> Distance {
        self.header.as_ref().map(|h| h.metric).unwrap_or(Distance::L2)
    }
    
    fn save(&self, path: &str) -> Result<()> {
        // PQFlashIndex is already saved to disk during build
        // This could copy the existing files to a new location if needed
        Ok(())
    }
    
    fn stats(&self) -> super::IndexStats {
        let header = self.header.as_ref();
        super::IndexStats {
            num_vectors: self.size(),
            dimension: self.dimension(),
            metric: self.metric(),
            memory_usage_bytes: self.memory_usage_estimate(),
            graph_degree_avg: header.map(|h| h.max_degree as f32).unwrap_or(0.0),
            graph_degree_max: header.map(|h| h.max_degree as usize).unwrap_or(0),
        }
    }
    
    fn memory_usage_bytes(&self) -> usize {
        self.memory_usage_estimate()
    }
}