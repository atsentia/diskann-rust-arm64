//! Model types for the external API

mod configuration;
pub use configuration::{IndexConfiguration, IndexWriteParameters, IndexWriteParametersBuilder, DiskIndexBuildParameters};

pub mod vertex {
    //! Vertex dimension constants
    
    /// Dimension 104 (aligned)
    pub const DIM_104: usize = 104;
    
    /// Dimension 128 (aligned)
    pub const DIM_128: usize = 128;
    
    /// Dimension 256 (aligned)
    pub const DIM_256: usize = 256;
}