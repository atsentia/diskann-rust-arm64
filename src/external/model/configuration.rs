//! Configuration types for index building and searching

use crate::external::Metric;

/// Index configuration - streamlined version
#[derive(Debug, Clone)]
pub struct IndexConfiguration {
    /// Index write parameters
    pub index_write_parameter: IndexWriteParameters,
    
    /// Distance metric
    pub dist_metric: Metric,
    
    /// Dimension of the raw data
    pub dim: usize,
    
    /// Aligned dimension - round up dim to the nearest multiple of 8
    pub aligned_dim: usize,
    
    /// Total number of points in given data set
    pub max_points: usize,
    
    /// Number of frozen points (for dynamic index)
    pub num_frozen_pts: usize,
    
    /// Use PQ distance calculation
    pub use_pq_dist: bool,
    
    /// Number of PQ chunks
    pub num_pq_chunks: usize,
    
    /// Use optimized product quantization
    pub use_opq: bool,
    
    /// Growth potential (1.2 = 20% growth)
    pub growth_potential: f32,
}

impl IndexConfiguration {
    /// Create a new index configuration
    pub fn new(
        dist_metric: Metric,
        dim: usize,
        aligned_dim: usize,
        max_points: usize,
        use_pq_dist: bool,
        num_frozen_pts: usize,
        use_opq: bool,
        num_pq_chunks: usize,
        growth_potential: f32,
        index_write_parameter: IndexWriteParameters,
    ) -> Self {
        Self {
            dist_metric,
            dim,
            aligned_dim,
            max_points,
            use_pq_dist,
            num_frozen_pts,
            use_opq,
            num_pq_chunks,
            growth_potential,
            index_write_parameter,
        }
    }
}

/// Index write parameters
#[derive(Debug, Clone, Copy)]
pub struct IndexWriteParameters {
    /// Search list size (L)
    pub search_list_size: u32,
    
    /// Max degree (R)
    pub max_degree: u32,
    
    /// Saturate graph
    pub saturate_graph: bool,
    
    /// Max occlusion size (C)
    pub max_occlusion_size: u32,
    
    /// Alpha parameter
    pub alpha: f32,
    
    /// Number of rounds
    pub num_rounds: u32,
    
    /// Number of threads (0 = use all)
    pub num_threads: u32,
    
    /// Number of frozen points
    pub num_frozen_points: u32,
}

impl Default for IndexWriteParameters {
    fn default() -> Self {
        Self {
            search_list_size: 100,
            max_degree: 64,
            saturate_graph: false,
            max_occlusion_size: 750,
            alpha: 1.2,
            num_rounds: 2,
            num_threads: 0,
            num_frozen_points: 0,
        }
    }
}

/// Builder for IndexWriteParameters
#[derive(Debug)]
pub struct IndexWriteParametersBuilder {
    params: IndexWriteParameters,
}

impl IndexWriteParametersBuilder {
    /// Create a new builder with required parameters
    pub fn new(search_list_size: u32, max_degree: u32) -> Self {
        let mut params = IndexWriteParameters::default();
        params.search_list_size = search_list_size;
        params.max_degree = max_degree;
        Self { params }
    }
    
    /// Set alpha parameter
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.params.alpha = alpha;
        self
    }
    
    /// Set saturate graph
    pub fn with_saturate_graph(mut self, saturate: bool) -> Self {
        self.params.saturate_graph = saturate;
        self
    }
    
    /// Set number of threads
    pub fn with_num_threads(mut self, threads: u32) -> Self {
        self.params.num_threads = threads;
        self
    }
    
    /// Set max occlusion size
    pub fn with_max_occlusion_size(mut self, size: u32) -> Self {
        self.params.max_occlusion_size = size;
        self
    }
    
    /// Set number of rounds
    pub fn with_num_rounds(mut self, rounds: u32) -> Self {
        self.params.num_rounds = rounds;
        self
    }
    
    /// Build the parameters
    pub fn build(self) -> IndexWriteParameters {
        self.params
    }
}

/// Disk index build parameters
#[derive(Debug, Clone)]
pub struct DiskIndexBuildParameters {
    /// Search list size for building
    pub search_list_size: u32,
    
    /// Build list size
    pub build_list_size: u32,
    
    /// Maximum degree
    pub max_degree: u32,
    
    /// Build PQ bytes
    pub build_pq_bytes: u32,
    
    /// Use OPQ
    pub use_opq: bool,
}