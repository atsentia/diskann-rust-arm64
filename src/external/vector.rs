//! Minimal vector crate types for Microsoft DiskANN compatibility
//!
//! This module provides the minimal types needed to match the Microsoft
//! DiskANN Rust API without requiring external dependencies.

use std::fmt;
use std::str::FromStr;

/// Distance metric enum matching Microsoft's vector crate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// L2 (Euclidean) distance
    L2,
    /// Cosine similarity
    Cosine,
    /// Inner product (dot product)
    InnerProduct,
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Metric::L2 => write!(f, "l2"),
            Metric::Cosine => write!(f, "cosine"),
            Metric::InnerProduct => write!(f, "inner_product"),
        }
    }
}

impl FromStr for Metric {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "l2" => Ok(Metric::L2),
            "cosine" => Ok(Metric::Cosine),
            "inner_product" | "innerproduct" | "ip" => Ok(Metric::InnerProduct),
            _ => Err(format!("Unknown metric: {}", s)),
        }
    }
}

/// Half precision float type
pub type Half = half::f16;

/// Trait for full precision distance calculation
pub trait FullPrecisionDistance<T, const N: usize> {
    fn distance(&self, other: &[T; N], metric: Metric) -> f32;
}

// Basic implementation for f32 arrays
impl<const N: usize> FullPrecisionDistance<f32, N> for [f32; N] {
    fn distance(&self, other: &[f32; N], metric: Metric) -> f32 {
        match metric {
            Metric::L2 => {
                self.iter()
                    .zip(other.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt()
            }
            Metric::Cosine => {
                let dot: f32 = self.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
                let norm_a: f32 = self.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 - (dot / (norm_a * norm_b))
            }
            Metric::InnerProduct => {
                -self.iter().zip(other.iter()).map(|(a, b)| a * b).sum::<f32>()
            }
        }
    }
}

// Implementation for Half (f16) arrays
impl<const N: usize> FullPrecisionDistance<Half, N> for [Half; N] {
    fn distance(&self, other: &[Half; N], metric: Metric) -> f32 {
        // Convert to f32 and calculate
        let self_f32: Vec<f32> = self.iter().map(|&x| x.to_f32()).collect();
        let other_f32: Vec<f32> = other.iter().map(|&x| x.to_f32()).collect();
        
        match metric {
            Metric::L2 => {
                self_f32.iter()
                    .zip(other_f32.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt()
            }
            Metric::Cosine => {
                let dot: f32 = self_f32.iter().zip(other_f32.iter()).map(|(a, b)| a * b).sum();
                let norm_a: f32 = self_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = other_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 - (dot / (norm_a * norm_b))
            }
            Metric::InnerProduct => {
                -self_f32.iter().zip(other_f32.iter()).map(|(a, b)| a * b).sum::<f32>()
            }
        }
    }
}