//! Implementation of index wrappers

use std::sync::Arc;
use parking_lot::RwLock;

use crate::{
    index::{Index, IndexBuilder, DynamicIndex, PQFlashIndex, PQFlashConfig},
    formats
};
use crate::external::{ANNResult, ANNError, IndexConfiguration, DiskIndexBuildParameters};
use super::{ANNInmemIndex, ANNDiskIndex};

/// Wrapper for in-memory index
pub struct InmemIndexWrapper<T> 
where T: Default + Copy + Sync + Send + Into<f32>
{
    index: Option<Arc<RwLock<crate::index::MemoryIndex>>>,
    dynamic_index: Option<Arc<RwLock<DynamicIndex>>>,
    builder: IndexBuilder,
    config: IndexConfiguration,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> InmemIndexWrapper<T>
where T: Default + Copy + Sync + Send + Into<f32>
{
    pub fn new(builder: IndexBuilder, config: IndexConfiguration) -> Self {
        Self {
            index: None,
            dynamic_index: None,
            builder,
            config,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Convert T vectors to f32
    fn convert_vectors(vectors: Vec<Vec<T>>) -> Vec<Vec<f32>> {
        vectors.into_iter()
            .map(|v| v.into_iter().map(|x| x.into()).collect())
            .collect()
    }
}

unsafe impl<T> Send for InmemIndexWrapper<T> where T: Default + Copy + Sync + Send + Into<f32> {}
unsafe impl<T> Sync for InmemIndexWrapper<T> where T: Default + Copy + Sync + Send + Into<f32> {}

impl<T> ANNInmemIndex<T> for InmemIndexWrapper<T>
where T: Default + Copy + Sync + Send + Into<f32> + 'static
{
    fn build(&mut self, filename: &str, num_points_to_load: usize) -> ANNResult<()> {
        // Load vectors from file
        let (vectors, _dim) = if filename.ends_with(".fvecs") {
            formats::read_fvecs(filename)?
        } else if filename.ends_with(".bin") || filename.ends_with(".fbin") {
            let vectors = formats::read_binary_vectors(filename, self.config.dim)?;
            (vectors, self.config.dim)
        } else {
            return Err(ANNError::InvalidParameter(format!("Unsupported file format: {}", filename)));
        };
        
        // Limit to requested number of points
        let vectors: Vec<Vec<f32>> = vectors.into_iter()
            .take(num_points_to_load)
            .collect();
        
        // Build the index
        let metric = match self.config.dist_metric {
            crate::external::Metric::L2 => crate::Distance::L2,
            crate::external::Metric::Cosine => crate::Distance::Cosine,
            crate::external::Metric::InnerProduct => crate::Distance::InnerProduct,
        };
        
        let index = crate::index::MemoryIndex::build(
            vectors,
            self.config.dim,
            metric,
            self.config.index_write_parameter.max_degree as usize,
            self.config.index_write_parameter.search_list_size as usize,
            self.config.index_write_parameter.alpha,
        )?;
        
        self.index = Some(Arc::new(RwLock::new(index)));
        Ok(())
    }
    
    fn save(&mut self, filename: &str) -> ANNResult<()> {
        if let Some(index) = &self.index {
            index.read().save(filename)?;
            Ok(())
        } else {
            Err(ANNError::IndexError("No index to save".to_string()))
        }
    }
    
    fn load(&mut self, filename: &str, _expected_num_points: usize) -> ANNResult<()> {
        // Load index from file
        let index = crate::index::MemoryIndex::load(filename)?;
        self.index = Some(Arc::new(RwLock::new(index)));
        Ok(())
    }
    
    fn insert(&mut self, filename: &str, num_points_to_insert: usize) -> ANNResult<()> {
        // For insert operations, we need to use DynamicIndex
        if self.dynamic_index.is_none() {
            // Convert existing index to dynamic if needed
            if let Some(index) = &self.index {
                let index_guard = index.read();
                // This is a simplified approach - in production you'd want to convert properly
                return Err(ANNError::NotImplemented("Converting static to dynamic index not implemented".to_string()));
            }
            
            // Create new dynamic index
            let metric = match self.config.dist_metric {
                crate::external::Metric::L2 => crate::Distance::L2,
                crate::external::Metric::Cosine => crate::Distance::Cosine,
                crate::external::Metric::InnerProduct => crate::Distance::InnerProduct,
            };
            let dynamic = DynamicIndex::new(
                self.config.dim,
                metric,
                self.config.index_write_parameter.max_degree as usize,
                self.config.index_write_parameter.search_list_size as usize,
                self.config.index_write_parameter.alpha,
            );
            self.dynamic_index = Some(Arc::new(RwLock::new(dynamic)));
        }
        
        // Load vectors to insert
        let (vectors, _) = if filename.ends_with(".fvecs") {
            formats::read_fvecs(filename)?
        } else {
            let vectors = formats::read_binary_vectors(filename, self.config.dim)?;
            (vectors, self.config.dim)
        };
        
        // Insert vectors
        if let Some(dynamic) = &self.dynamic_index {
            let mut index = dynamic.write();
            for vector in vectors.iter().take(num_points_to_insert) {
                let _ = index.insert(vector.clone(), vec![]);
            }
        }
        
        Ok(())
    }
    
    fn search(&self, query: &[T], k_value: usize, _l_value: u32, indices: &mut [u32]) -> ANNResult<u32> {
        // Convert query to f32
        let query_f32: Vec<f32> = query.iter().map(|&x| x.into()).collect();
        
        // Search in whichever index we have
        let results = if let Some(index) = &self.index {
            index.read().search(&query_f32, k_value)?
        } else if let Some(dynamic) = &self.dynamic_index {
            dynamic.read().search(&query_f32, k_value)?
        } else {
            return Err(ANNError::IndexError("No index available".to_string()));
        };
        
        // Copy results to output array
        let num_found = results.len().min(indices.len());
        for (i, (id, _dist)) in results.iter().take(num_found).enumerate() {
            indices[i] = *id as u32;
        }
        
        Ok(num_found as u32)
    }
    
    fn soft_delete(&mut self, vertex_ids_to_delete: Vec<u32>, _num_points_to_delete: usize) -> ANNResult<()> {
        if let Some(dynamic) = &self.dynamic_index {
            let mut index = dynamic.write();
            for id in vertex_ids_to_delete {
                index.delete(id as usize);
            }
            Ok(())
        } else {
            Err(ANNError::NotImplemented("Delete only supported on dynamic index".to_string()))
        }
    }
}

/// Wrapper for disk index
pub struct DiskIndexWrapper<T>
where T: Default + Copy + Sync + Send + Into<f32>
{
    index: Option<PQFlashIndex>,
    build_params: Option<DiskIndexBuildParameters>,
    config: IndexConfiguration,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> DiskIndexWrapper<T>
where T: Default + Copy + Sync + Send + Into<f32>
{
    pub fn new(
        build_params: Option<DiskIndexBuildParameters>,
        config: IndexConfiguration,
    ) -> Self {
        Self {
            index: None,
            build_params,
            config,
            _phantom: std::marker::PhantomData,
        }
    }
}

unsafe impl<T> Send for DiskIndexWrapper<T> where T: Default + Copy + Sync + Send + Into<f32> {}
unsafe impl<T> Sync for DiskIndexWrapper<T> where T: Default + Copy + Sync + Send + Into<f32> {}

impl<T> ANNDiskIndex<T> for DiskIndexWrapper<T>
where T: Default + Copy + Sync + Send + Into<f32> + 'static
{
    fn build(&mut self, data_file: &str, codebook_prefix: &str) -> ANNResult<()> {
        // Load vectors
        let (vectors, _) = if data_file.ends_with(".fvecs") {
            formats::read_fvecs(data_file)?
        } else {
            let vectors = formats::read_binary_vectors(data_file, self.config.dim)?;
            (vectors, self.config.dim)
        };
        
        // Map metric
        let metric = match self.config.dist_metric {
            crate::external::Metric::L2 => crate::Distance::L2,
            crate::external::Metric::Cosine => crate::Distance::Cosine,
            crate::external::Metric::InnerProduct => crate::Distance::InnerProduct,
        };
        
        // Create PQ config
        let pq_config = PQFlashConfig {
            max_degree: self.config.index_write_parameter.max_degree as usize,
            search_list_size: self.config.index_write_parameter.search_list_size as usize,
            alpha: self.config.index_write_parameter.alpha,
            pq_params: crate::index::disk::PQParams {
                num_chunks: self.config.num_pq_chunks.max(1),
                bits_per_chunk: 8, // Default to 8 bits
            },
            num_threads: self.config.index_write_parameter.num_threads as usize,
            use_reorder_data: true, // Default to using reorder data
            beam_width: 4, // Default beam width
        };
        
        // Build index with metric
        let index_path = format!("{}.pq", codebook_prefix);
        let mut index = PQFlashIndex::new(self.config.dim, metric, pq_config);
        index.build_and_save(&vectors, &index_path)?;
        index.load(&index_path)?;
        self.index = Some(index);
        
        Ok(())
    }
    
    fn load(&mut self, index_path: &str) -> ANNResult<()> {
        let metric = match self.config.dist_metric {
            crate::external::Metric::L2 => crate::Distance::L2,
            crate::external::Metric::Cosine => crate::Distance::Cosine,
            crate::external::Metric::InnerProduct => crate::Distance::InnerProduct,
        };
        
        let mut index = PQFlashIndex::new(
            self.config.dim,
            metric,
            PQFlashConfig {
                max_degree: self.config.index_write_parameter.max_degree as usize,
                search_list_size: self.config.index_write_parameter.search_list_size as usize,
                alpha: self.config.index_write_parameter.alpha,
                pq_params: crate::index::disk::PQParams {
                    num_chunks: self.config.num_pq_chunks.max(1),
                    bits_per_chunk: 8,
                },
                num_threads: self.config.index_write_parameter.num_threads as usize,
                use_reorder_data: true,
                beam_width: 4,
            }
        );
        index.load(index_path)?;
        self.index = Some(index);
        Ok(())
    }
    
    fn search(&mut self, query: &[T], k_value: usize, indices: &mut [u32], distances: &mut [f32]) -> ANNResult<u32> {
        // Convert query to f32
        let query_f32: Vec<f32> = query.iter().map(|&x| x.into()).collect();
        
        if let Some(index) = &mut self.index {
            let search_list_size = k_value * 10; // Default search parameter
            let (results, _stats) = index.search(&query_f32, k_value, search_list_size)?;
            
            // Copy results
            let num_found = results.len().min(indices.len()).min(distances.len());
            for (i, (id, dist)) in results.iter().take(num_found).enumerate() {
                indices[i] = *id as u32;
                distances[i] = *dist;
            }
            
            Ok(num_found as u32)
        } else {
            Err(ANNError::IndexError("No index loaded".to_string()))
        }
    }
}