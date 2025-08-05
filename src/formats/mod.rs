//! File format support for various vector formats
//!
//! This module provides readers and writers for common vector file formats
//! including fvecs, bvecs, ivecs, and binary formats.

use crate::{Result, Error};
use crate::types::{VectorType, VectorElement};
use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom, BufReader, BufWriter};
use std::path::Path;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Common vector file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorFormat {
    /// Float vectors (.fvecs) - 4 bytes per element
    Fvecs,
    /// Byte vectors (.bvecs) - 1 byte per element  
    Bvecs,
    /// Integer vectors (.ivecs) - 4 bytes per element
    Ivecs,
    /// Binary format with header - flexible types
    Binary,
}

impl VectorFormat {
    /// Detect format from file extension
    pub fn from_path<P: AsRef<Path>>(path: P) -> Option<Self> {
        let ext = path.as_ref()
            .extension()?
            .to_str()?
            .to_lowercase();
        
        match ext.as_str() {
            "fvecs" => Some(VectorFormat::Fvecs),
            "bvecs" => Some(VectorFormat::Bvecs),
            "ivecs" => Some(VectorFormat::Ivecs),
            "bin" => Some(VectorFormat::Binary),
            _ => None,
        }
    }
}

/// Read vectors from fvecs format
pub fn read_fvecs<P: AsRef<Path>>(path: P) -> Result<(Vec<Vec<f32>>, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    let mut dimension = None;
    
    loop {
        // Read dimension
        let dim = match reader.read_u32::<LittleEndian>() {
            Ok(d) => d as usize,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        
        // Verify consistent dimension
        if let Some(expected_dim) = dimension {
            if dim != expected_dim {
                return Err(Error::InvalidParameter(
                    format!("Inconsistent dimensions: {} vs {}", dim, expected_dim)
                ));
            }
        } else {
            dimension = Some(dim);
        }
        
        // Read vector data
        let mut vector = vec![0.0f32; dim];
        for i in 0..dim {
            vector[i] = reader.read_f32::<LittleEndian>()?;
        }
        
        vectors.push(vector);
    }
    
    let dim = dimension.unwrap_or(0);
    Ok((vectors, dim))
}

/// Write vectors to fvecs format
pub fn write_fvecs<P: AsRef<Path>>(path: P, vectors: &[Vec<f32>]) -> Result<()> {
    if vectors.is_empty() {
        return Ok(());
    }
    
    let dimension = vectors[0].len();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    for vector in vectors {
        if vector.len() != dimension {
            return Err(Error::DimensionMismatch {
                expected: dimension,
                actual: vector.len(),
            });
        }
        
        // Write dimension
        writer.write_u32::<LittleEndian>(dimension as u32)?;
        
        // Write vector data
        for &value in vector {
            writer.write_f32::<LittleEndian>(value)?;
        }
    }
    
    writer.flush()?;
    Ok(())
}

/// Read vectors from bvecs format
pub fn read_bvecs<P: AsRef<Path>>(path: P) -> Result<(Vec<Vec<u8>>, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    let mut dimension = None;
    
    loop {
        // Read dimension
        let dim = match reader.read_u32::<LittleEndian>() {
            Ok(d) => d as usize,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        
        // Verify consistent dimension
        if let Some(expected_dim) = dimension {
            if dim != expected_dim {
                return Err(Error::InvalidParameter(
                    format!("Inconsistent dimensions: {} vs {}", dim, expected_dim)
                ));
            }
        } else {
            dimension = Some(dim);
        }
        
        // Read vector data
        let mut vector = vec![0u8; dim];
        reader.read_exact(&mut vector)?;
        
        vectors.push(vector);
    }
    
    let dim = dimension.unwrap_or(0);
    Ok((vectors, dim))
}

/// Write vectors to bvecs format
pub fn write_bvecs<P: AsRef<Path>>(path: P, vectors: &[Vec<u8>]) -> Result<()> {
    if vectors.is_empty() {
        return Ok(());
    }
    
    let dimension = vectors[0].len();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    for vector in vectors {
        if vector.len() != dimension {
            return Err(Error::DimensionMismatch {
                expected: dimension,
                actual: vector.len(),
            });
        }
        
        // Write dimension
        writer.write_u32::<LittleEndian>(dimension as u32)?;
        
        // Write vector data
        writer.write_all(vector)?;
    }
    
    writer.flush()?;
    Ok(())
}

/// Read vectors from ivecs format
pub fn read_ivecs<P: AsRef<Path>>(path: P) -> Result<(Vec<Vec<i32>>, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    let mut dimension = None;
    
    loop {
        // Read dimension
        let dim = match reader.read_u32::<LittleEndian>() {
            Ok(d) => d as usize,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        
        // Verify consistent dimension
        if let Some(expected_dim) = dimension {
            if dim != expected_dim {
                return Err(Error::InvalidParameter(
                    format!("Inconsistent dimensions: {} vs {}", dim, expected_dim)
                ));
            }
        } else {
            dimension = Some(dim);
        }
        
        // Read vector data
        let mut vector = vec![0i32; dim];
        for i in 0..dim {
            vector[i] = reader.read_i32::<LittleEndian>()?;
        }
        
        vectors.push(vector);
    }
    
    let dim = dimension.unwrap_or(0);
    Ok((vectors, dim))
}

/// Write vectors to ivecs format
pub fn write_ivecs<P: AsRef<Path>>(path: P, vectors: &[Vec<i32>]) -> Result<()> {
    if vectors.is_empty() {
        return Ok(());
    }
    
    let dimension = vectors[0].len();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    for vector in vectors {
        if vector.len() != dimension {
            return Err(Error::DimensionMismatch {
                expected: dimension,
                actual: vector.len(),
            });
        }
        
        // Write dimension
        writer.write_u32::<LittleEndian>(dimension as u32)?;
        
        // Write vector data
        for &value in vector {
            writer.write_i32::<LittleEndian>(value)?;
        }
    }
    
    writer.flush()?;
    Ok(())
}

/// Binary format header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BinaryHeader {
    pub num_vectors: u32,
    pub dimension: u32,
    pub vector_type: u32,
    pub reserved: [u32; 5],
}

impl BinaryHeader {
    pub const SIZE: usize = 32; // 8 * 4 bytes
    
    pub fn new(num_vectors: u32, dimension: u32, vector_type: VectorType) -> Self {
        Self {
            num_vectors,
            dimension,
            vector_type: vector_type as u32,
            reserved: [0; 5],
        }
    }
    
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        let mut cursor = &mut bytes[..];
        
        cursor.write_u32::<LittleEndian>(self.num_vectors).unwrap();
        cursor.write_u32::<LittleEndian>(self.dimension).unwrap();
        cursor.write_u32::<LittleEndian>(self.vector_type).unwrap();
        for &val in &self.reserved {
            cursor.write_u32::<LittleEndian>(val).unwrap();
        }
        
        bytes
    }
    
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Invalid header size"
            )));
        }
        
        let mut cursor = &bytes[..Self::SIZE];
        let num_vectors = cursor.read_u32::<LittleEndian>()?;
        let dimension = cursor.read_u32::<LittleEndian>()?;
        let vector_type = cursor.read_u32::<LittleEndian>()?;
        
        let mut reserved = [0u32; 5];
        for i in 0..5 {
            reserved[i] = cursor.read_u32::<LittleEndian>()?;
        }
        
        Ok(Self {
            num_vectors,
            dimension,
            vector_type,
            reserved,
        })
    }
}

/// Read vectors from binary format
pub fn read_binary<P: AsRef<Path>>(path: P) -> Result<(Vec<Vec<f32>>, usize, VectorType)> {
    let mut file = File::open(path)?;
    
    // Read header
    let mut header_bytes = [0u8; BinaryHeader::SIZE];
    file.read_exact(&mut header_bytes)?;
    let header = BinaryHeader::from_bytes(&header_bytes)?;
    
    let num_vectors = header.num_vectors as usize;
    let dimension = header.dimension as usize;
    let vector_type = match header.vector_type {
        0 => VectorType::Float32,
        1 => VectorType::Float16,
        2 => VectorType::Int8,
        3 => VectorType::UInt8,
        _ => return Err(Error::InvalidParameter("Unknown vector type".to_owned()).into()),
    };
    
    // Read vectors based on type
    let mut vectors = Vec::with_capacity(num_vectors);
    let element_size = vector_type.size();
    let vector_bytes = dimension * element_size;
    
    for _ in 0..num_vectors {
        let mut buffer = vec![0u8; vector_bytes];
        file.read_exact(&mut buffer)?;
        
        // Convert to f32
        let vector = match vector_type {
            VectorType::Float32 => {
                buffer.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            VectorType::Float16 => {
                buffer.chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect()
            }
            VectorType::Int8 => {
                buffer.iter()
                    .map(|&b| i8::from_le_bytes([b]) as f32)
                    .collect()
            }
            VectorType::UInt8 => {
                buffer.iter()
                    .map(|&b| b as f32)
                    .collect()
            }
        };
        
        vectors.push(vector);
    }
    
    Ok((vectors, dimension, vector_type))
}

/// Write vectors to binary format
pub fn write_binary<P: AsRef<Path>>(
    path: P,
    vectors: &[Vec<f32>],
    vector_type: VectorType,
) -> Result<()> {
    if vectors.is_empty() {
        return Ok(());
    }
    
    let num_vectors = vectors.len() as u32;
    let dimension = vectors[0].len() as u32;
    
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // Write header
    let header = BinaryHeader::new(num_vectors, dimension, vector_type);
    writer.write_all(&header.to_bytes())?;
    
    // Write vectors
    for vector in vectors {
        if vector.len() != dimension as usize {
            return Err(Error::DimensionMismatch {
                expected: dimension as usize,
                actual: vector.len(),
            }.into());
        }
        
        // Convert and write based on type
        match vector_type {
            VectorType::Float32 => {
                for &value in vector {
                    writer.write_f32::<LittleEndian>(value)?;
                }
            }
            VectorType::Float16 => {
                for &value in vector {
                    let f16_val = half::f16::from_f32(value);
                    writer.write_u16::<LittleEndian>(f16_val.to_bits())?;
                }
            }
            VectorType::Int8 => {
                for &value in vector {
                    writer.write_i8(value.round().clamp(-128.0, 127.0) as i8)?;
                }
            }
            VectorType::UInt8 => {
                for &value in vector {
                    writer.write_u8(value.round().clamp(0.0, 255.0) as u8)?;
                }
            }
        }
    }
    
    writer.flush()?;
    Ok(())
}

/// Convert between vector formats
pub fn convert_format<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_path: P1,
    output_path: P2,
    output_format: VectorFormat,
) -> Result<()> {
    // Detect input format
    let input_format = VectorFormat::from_path(&input_path)
        .ok_or_else(|| Error::InvalidParameter("Unknown input format".to_owned()).into())?;
    
    // Read vectors
    let (vectors, _) = match input_format {
        VectorFormat::Fvecs => read_fvecs(input_path)?,
        VectorFormat::Bvecs => {
            let (byte_vecs, dim) = read_bvecs(input_path)?;
            let float_vecs: Vec<Vec<f32>> = byte_vecs.into_iter()
                .map(|v| v.into_iter().map(|b| b as f32).collect())
                .collect();
            (float_vecs, dim)
        }
        VectorFormat::Ivecs => {
            let (int_vecs, dim) = read_ivecs(input_path)?;
            let float_vecs: Vec<Vec<f32>> = int_vecs.into_iter()
                .map(|v| v.into_iter().map(|i| i as f32).collect())
                .collect();
            (float_vecs, dim)
        }
        VectorFormat::Binary => {
            let (vecs, dim, _) = read_binary(input_path)?;
            (vecs, dim)
        }
    };
    
    // Write vectors
    match output_format {
        VectorFormat::Fvecs => write_fvecs(output_path, &vectors)?,
        VectorFormat::Bvecs => {
            let byte_vecs: Vec<Vec<u8>> = vectors.into_iter()
                .map(|v| v.into_iter()
                    .map(|f| f.round().clamp(0.0, 255.0) as u8)
                    .collect())
                .collect();
            write_bvecs(output_path, &byte_vecs)?
        }
        VectorFormat::Ivecs => {
            let int_vecs: Vec<Vec<i32>> = vectors.into_iter()
                .map(|v| v.into_iter()
                    .map(|f| f.round() as i32)
                    .collect())
                .collect();
            write_ivecs(output_path, &int_vecs)?
        }
        VectorFormat::Binary => write_binary(output_path, &vectors, VectorType::Float32)?,
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_fvecs_round_trip() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        let file = NamedTempFile::new().unwrap();
        write_fvecs(file.path(), &vectors).unwrap();
        
        let (read_vectors, dim) = read_fvecs(file.path()).unwrap();
        assert_eq!(dim, 3);
        assert_eq!(read_vectors, vectors);
    }
    
    #[test]
    fn test_binary_format() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];
        
        let file = NamedTempFile::new().unwrap();
        write_binary(file.path(), &vectors, VectorType::Float16).unwrap();
        
        let (read_vectors, dim, vtype) = read_binary(file.path()).unwrap();
        assert_eq!(dim, 4);
        assert_eq!(vtype, VectorType::Float16);
        assert_eq!(read_vectors.len(), 2);
        
        // Check values are close (float16 has less precision)
        for (orig, read) in vectors.iter().zip(read_vectors.iter()) {
            for (o, r) in orig.iter().zip(read.iter()) {
                assert!((o - r).abs() < 0.01);
            }
        }
    }
}