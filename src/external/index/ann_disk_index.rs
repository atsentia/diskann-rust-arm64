//! Disk index abstraction

use crate::external::{ANNResult, IndexConfiguration, DiskIndexBuildParameters};
use super::index_impl::DiskIndexWrapper;

/// ANN disk index trait
pub trait ANNDiskIndex<T>: Sync + Send
where T: Default + Copy + Sync + Send + Into<f32>
{
    /// Build index from data file
    fn build(&mut self, data_file: &str, codebook_prefix: &str) -> ANNResult<()>;
    
    /// Load index from disk
    fn load(&mut self, index_path: &str) -> ANNResult<()>;
    
    /// Search the index
    fn search(&mut self, query: &[T], k_value: usize, indices: &mut [u32], distances: &mut [f32]) -> ANNResult<u32>;
}

/// Storage trait placeholder (simplified for now)
pub struct DiskIndexStorage<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for DiskIndexStorage<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Create a disk index based on configuration
pub fn create_disk_index<T>(
    disk_build_param: Option<DiskIndexBuildParameters>,
    config: IndexConfiguration,
    _storage: DiskIndexStorage<T>,
) -> ANNResult<Box<dyn ANNDiskIndex<T>>>
where
    T: Default + Copy + Sync + Send + Into<f32> + 'static,
{
    let wrapper = DiskIndexWrapper::new(disk_build_param, config);
    Ok(Box::new(wrapper))
}