//! Utilities for the external API

use crate::external::ANNResult;
use std::fs::File;
use byteorder::{LittleEndian, ReadBytesExt};

/// Load metadata (num points, dimension) from binary file
pub fn load_metadata_from_file(path: &str) -> ANNResult<(usize, usize)> {
    let mut file = File::open(path)?;
    
    // Try to read as binary format with header
    let mut num_points = file.read_u32::<LittleEndian>()? as usize;
    let mut dimension = file.read_u32::<LittleEndian>()? as usize;
    
    // Sanity check - if these look like data rather than metadata,
    // assume it's a raw binary file without header
    if num_points > 100_000_000 || dimension > 10_000 {
        // Reset and calculate from file size
        let metadata = std::fs::metadata(path)?;
        let file_size = metadata.len() as usize;
        
        // Assume float32 data
        let total_floats = file_size / 4;
        
        // Try common dimensions
        for dim in [128, 256, 512, 768, 960, 1024, 1536, 2048] {
            if total_floats % dim == 0 {
                num_points = total_floats / dim;
                dimension = dim;
                break;
            }
        }
    }
    
    Ok((num_points, dimension))
}

/// Round up to nearest multiple
pub fn round_up(value: u64, multiple: u64) -> u64 {
    ((value + multiple - 1) / multiple) * multiple
}

/// Timer for benchmarking
pub struct Timer {
    start: std::time::Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }
    
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
}