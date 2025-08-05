//! Data type support for various vector formats
//!
//! This module provides support for different data types including
//! int8, uint8, float16, and float32.

use crate::{Result, Error};
use half::f16;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;

/// Supported vector element types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorType {
    Float32,
    Float16,
    Int8,
    UInt8,
}

impl VectorType {
    /// Get the size in bytes of this type
    pub fn size(&self) -> usize {
        match self {
            VectorType::Float32 => 4,
            VectorType::Float16 => 2,
            VectorType::Int8 | VectorType::UInt8 => 1,
        }
    }
    
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "float" | "float32" | "f32" => Some(VectorType::Float32),
            "float16" | "f16" => Some(VectorType::Float16),
            "int8" | "i8" => Some(VectorType::Int8),
            "uint8" | "u8" => Some(VectorType::UInt8),
            _ => None,
        }
    }
}

/// Trait for vector element types
pub trait VectorElement: Copy + Debug + Send + Sync + 'static {
    /// Convert to f32 for distance calculations
    fn to_f32(self) -> f32;
    
    /// Convert from f32
    fn from_f32(value: f32) -> Self;
    
    /// Get the vector type enum
    fn vector_type() -> VectorType;
    
    /// Scale value to a different range
    fn scale(self, scale: f32, offset: f32) -> Self {
        Self::from_f32(self.to_f32() * scale + offset)
    }
}

impl VectorElement for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    
    #[inline]
    fn from_f32(value: f32) -> Self {
        value
    }
    
    fn vector_type() -> VectorType {
        VectorType::Float32
    }
}

impl VectorElement for f16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    
    #[inline]
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }
    
    fn vector_type() -> VectorType {
        VectorType::Float16
    }
}

impl VectorElement for i8 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    
    #[inline]
    fn from_f32(value: f32) -> Self {
        value.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8
    }
    
    fn vector_type() -> VectorType {
        VectorType::Int8
    }
}

impl VectorElement for u8 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    
    #[inline]
    fn from_f32(value: f32) -> Self {
        value.round().clamp(0.0, u8::MAX as f32) as u8
    }
    
    fn vector_type() -> VectorType {
        VectorType::UInt8
    }
}

/// Generic vector that can hold different element types
#[derive(Debug, Clone)]
pub enum Vector {
    Float32(Vec<f32>),
    Float16(Vec<f16>),
    Int8(Vec<i8>),
    UInt8(Vec<u8>),
}

impl Vector {
    /// Create a new vector of the specified type
    pub fn new(vector_type: VectorType, dimension: usize) -> Self {
        match vector_type {
            VectorType::Float32 => Vector::Float32(vec![0.0; dimension]),
            VectorType::Float16 => Vector::Float16(vec![f16::ZERO; dimension]),
            VectorType::Int8 => Vector::Int8(vec![0; dimension]),
            VectorType::UInt8 => Vector::UInt8(vec![0; dimension]),
        }
    }
    
    /// Get the vector type
    pub fn vector_type(&self) -> VectorType {
        match self {
            Vector::Float32(_) => VectorType::Float32,
            Vector::Float16(_) => VectorType::Float16,
            Vector::Int8(_) => VectorType::Int8,
            Vector::UInt8(_) => VectorType::UInt8,
        }
    }
    
    /// Get the dimension
    pub fn dimension(&self) -> usize {
        match self {
            Vector::Float32(v) => v.len(),
            Vector::Float16(v) => v.len(),
            Vector::Int8(v) => v.len(),
            Vector::UInt8(v) => v.len(),
        }
    }
    
    /// Convert to f32 vector for distance calculations
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Vector::Float32(v) => v.clone(),
            Vector::Float16(v) => v.iter().map(|&x| x.to_f32()).collect(),
            Vector::Int8(v) => v.iter().map(|&x| x as f32).collect(),
            Vector::UInt8(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }
    
    /// Create from f32 vector
    pub fn from_f32_vec(data: Vec<f32>, vector_type: VectorType) -> Self {
        match vector_type {
            VectorType::Float32 => Vector::Float32(data),
            VectorType::Float16 => {
                Vector::Float16(data.iter().map(|&x| f16::from_f32(x)).collect())
            }
            VectorType::Int8 => {
                Vector::Int8(data.iter().map(|&x| i8::from_f32(x)).collect())
            }
            VectorType::UInt8 => {
                Vector::UInt8(data.iter().map(|&x| u8::from_f32(x)).collect())
            }
        }
    }
    
    /// Get raw bytes for storage
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Vector::Float32(v) => bytemuck::cast_slice(v),
            Vector::Float16(v) => bytemuck::cast_slice(v),
            Vector::Int8(v) => bytemuck::cast_slice(v),
            Vector::UInt8(v) => bytemuck::cast_slice(v),
        }
    }
}

/// Scale parameters for quantized types
#[derive(Debug, Clone, Copy)]
pub struct ScaleParams {
    pub scale: f32,
    pub offset: f32,
}

impl ScaleParams {
    /// Create new scale parameters
    pub fn new(scale: f32, offset: f32) -> Self {
        Self { scale, offset }
    }
    
    /// Identity scaling (no change)
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            offset: 0.0,
        }
    }
    
    /// Compute scale parameters to map [min, max] to [target_min, target_max]
    pub fn compute(min: f32, max: f32, target_min: f32, target_max: f32) -> Self {
        let scale = (target_max - target_min) / (max - min);
        let offset = target_min - min * scale;
        Self { scale, offset }
    }
    
    /// Apply scaling
    pub fn apply(&self, value: f32) -> f32 {
        value * self.scale + self.offset
    }
    
    /// Apply inverse scaling
    pub fn apply_inverse(&self, value: f32) -> f32 {
        (value - self.offset) / self.scale
    }
}

/// Quantization utilities
pub mod quantize {
    use super::*;
    
    /// Quantize f32 vector to int8 with automatic scaling
    pub fn f32_to_int8(data: &[f32]) -> (Vec<i8>, ScaleParams) {
        let (min, max) = data.iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &x| {
                (min.min(x), max.max(x))
            });
        
        let params = ScaleParams::compute(min, max, -127.0, 127.0);
        
        let quantized: Vec<i8> = data.iter()
            .map(|&x| i8::from_f32(params.apply(x)))
            .collect();
        
        (quantized, params)
    }
    
    /// Quantize f32 vector to uint8 with automatic scaling
    pub fn f32_to_uint8(data: &[f32]) -> (Vec<u8>, ScaleParams) {
        let (min, max) = data.iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &x| {
                (min.min(x), max.max(x))
            });
        
        let params = ScaleParams::compute(min, max, 0.0, 255.0);
        
        let quantized: Vec<u8> = data.iter()
            .map(|&x| u8::from_f32(params.apply(x)))
            .collect();
        
        (quantized, params)
    }
    
    /// Dequantize int8 to f32 with scaling
    pub fn int8_to_f32(data: &[i8], params: &ScaleParams) -> Vec<f32> {
        data.iter()
            .map(|&x| params.apply_inverse(x as f32))
            .collect()
    }
    
    /// Dequantize uint8 to f32 with scaling
    pub fn uint8_to_f32(data: &[u8], params: &ScaleParams) -> Vec<f32> {
        data.iter()
            .map(|&x| params.apply_inverse(x as f32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_vector_element_conversions() {
        // Test f32
        assert_eq!(f32::from_f32(3.14), 3.14);
        assert_eq!(3.14f32.to_f32(), 3.14);
        
        // Test f16
        let f16_val = f16::from_f32(3.14);
        assert_relative_eq!(f16_val.to_f32(), 3.14, epsilon = 0.01);
        
        // Test i8
        assert_eq!(i8::from_f32(42.7), 43);
        assert_eq!(i8::from_f32(-128.5), -128);
        assert_eq!(i8::from_f32(200.0), 127); // Clamping
        
        // Test u8
        assert_eq!(u8::from_f32(42.7), 43);
        assert_eq!(u8::from_f32(-10.0), 0); // Clamping
        assert_eq!(u8::from_f32(300.0), 255); // Clamping
    }
    
    #[test]
    fn test_scale_params() {
        let params = ScaleParams::compute(-1.0, 1.0, 0.0, 255.0);
        
        assert_relative_eq!(params.apply(-1.0), 0.0, epsilon = 0.01);
        assert_relative_eq!(params.apply(0.0), 127.5, epsilon = 0.01);
        assert_relative_eq!(params.apply(1.0), 255.0, epsilon = 0.01);
        
        // Test inverse
        assert_relative_eq!(params.apply_inverse(0.0), -1.0, epsilon = 0.01);
        assert_relative_eq!(params.apply_inverse(255.0), 1.0, epsilon = 0.01);
    }
    
    #[test]
    fn test_quantization() {
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        
        // Test int8 quantization
        let (quantized_i8, params_i8) = quantize::f32_to_int8(&data);
        let dequantized_i8 = quantize::int8_to_f32(&quantized_i8, &params_i8);
        
        for (orig, deq) in data.iter().zip(dequantized_i8.iter()) {
            assert_relative_eq!(orig, deq, epsilon = 0.02);
        }
        
        // Test uint8 quantization
        let (quantized_u8, params_u8) = quantize::f32_to_uint8(&data);
        let dequantized_u8 = quantize::uint8_to_f32(&quantized_u8, &params_u8);
        
        for (orig, deq) in data.iter().zip(dequantized_u8.iter()) {
            assert_relative_eq!(orig, deq, epsilon = 0.02);
        }
    }
}