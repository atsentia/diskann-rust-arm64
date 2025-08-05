//! Utility functions for DiskANN
//!
//! This module provides common utilities used throughout the library.

pub mod metrics;
pub mod aligned;

use std::fs::File;
use std::io::{BufReader, Read};
use crate::Result;

/// Load vectors from a binary file
pub fn load_vectors_from_file(path: &str) -> Result<(Vec<Vec<f32>>, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    
    // Read header: num_vectors and dimension
    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    
    let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let dimension = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    // Read vectors
    let mut vectors = Vec::with_capacity(num_vectors);
    let mut buffer = vec![0u8; dimension * 4];
    
    for _ in 0..num_vectors {
        reader.read_exact(&mut buffer)?;
        let vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        vectors.push(vector);
    }
    
    Ok((vectors, dimension))
}

/// Save vectors to a binary file
pub fn save_vectors_to_file(path: &str, vectors: &[Vec<f32>], dimension: usize) -> Result<()> {
    use std::io::Write;
    
    let file = File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    
    // Write header
    writer.write_all(&(vectors.len() as u32).to_le_bytes())?;
    writer.write_all(&(dimension as u32).to_le_bytes())?;
    
    // Write vectors
    for vector in vectors {
        for &value in vector {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    
    writer.flush()?;
    Ok(())
}

/// Generate random vectors for testing
#[cfg(test)]
pub fn generate_random_vectors(num_vectors: usize, dimension: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..num_vectors)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}