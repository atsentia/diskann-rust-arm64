//! In-memory index abstraction

use crate::external::{ANNResult, IndexConfiguration};
use super::index_impl::InmemIndexWrapper;

/// ANN in-memory index trait
pub trait ANNInmemIndex<T>: Sync + Send 
where T: Default + Copy + Sync + Send + Into<f32>
{
    /// Build index from file
    fn build(&mut self, filename: &str, num_points_to_load: usize) -> ANNResult<()>;
    
    /// Save index to file
    fn save(&mut self, filename: &str) -> ANNResult<()>;
    
    /// Load index from file
    fn load(&mut self, filename: &str, expected_num_points: usize) -> ANNResult<()>;
    
    /// Insert points from file
    fn insert(&mut self, filename: &str, num_points_to_insert: usize) -> ANNResult<()>;
    
    /// Search for k nearest neighbors
    fn search(&self, query: &[T], k_value: usize, l_value: u32, indices: &mut [u32]) -> ANNResult<u32>;
    
    /// Soft delete vertices
    fn soft_delete(&mut self, vertex_ids_to_delete: Vec<u32>, num_points_to_delete: usize) -> ANNResult<()>;
}

/// Create an in-memory index based on configuration
pub fn create_inmem_index<T>(config: IndexConfiguration) -> ANNResult<Box<dyn ANNInmemIndex<T>>>
where
    T: Default + Copy + Sync + Send + Into<f32> + 'static,
{
    // Map Microsoft Metric to our Distance enum
    let distance = match config.dist_metric {
        crate::external::Metric::L2 => crate::Distance::L2,
        crate::external::Metric::Cosine => crate::Distance::Cosine,
        crate::external::Metric::InnerProduct => crate::Distance::InnerProduct,
    };
    
    // Create our internal index using the configuration
    let index_builder = crate::IndexBuilder::new()
        .dimensions(config.dim)
        .metric(distance)
        .max_degree(config.index_write_parameter.max_degree as usize)
        .search_list_size(config.index_write_parameter.search_list_size as usize)
        .alpha(config.index_write_parameter.alpha);
    
    // For now, we'll create an empty index that can be built later
    let wrapper = InmemIndexWrapper::new(index_builder, config);
    
    Ok(Box::new(wrapper))
}