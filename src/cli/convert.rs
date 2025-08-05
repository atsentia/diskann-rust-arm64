//! Format conversion command
//!
//! This module provides utilities for converting between different vector
//! formats and data types with optional quantization.

use clap::Args;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::*;
use crate::types::{VectorType, QuantizationParams};

#[derive(Args)]
pub struct ConvertArgs {
    /// Input file path
    #[arg(short, long)]
    pub input: PathBuf,
    
    /// Output file path
    #[arg(short, long)]
    pub output: PathBuf,
    
    /// Input format (auto, fvecs, bvecs, ivecs, bin)
    #[arg(long, default_value = "auto")]
    pub input_format: String,
    
    /// Output format (fvecs, bvecs, ivecs, bin)
    #[arg(long)]
    pub output_format: String,
    
    /// Input dimension (required for binary format)
    #[arg(long)]
    pub input_dimension: Option<usize>,
    
    /// Output data type (float32, float16, int8, uint8)
    #[arg(long, default_value = "float32")]
    pub output_type: String,
    
    /// Quantization method for type conversion (minmax, standard)
    #[arg(long, default_value = "minmax")]
    pub quantization: String,
    
    /// Number of vectors to convert (all if not specified)
    #[arg(long)]
    pub limit: Option<usize>,
    
    /// Skip first N vectors
    #[arg(long, default_value = "0")]
    pub skip: usize,
    
    /// Normalize vectors to unit length
    #[arg(long)]
    pub normalize: bool,
    
    /// Center vectors (subtract mean)
    #[arg(long)]
    pub center: bool,
    
    /// Show conversion statistics
    #[arg(long)]
    pub stats: bool,
}

pub fn run(args: ConvertArgs, cli: &crate::Cli) -> diskann::Result<()> {
    if !cli.no_progress {
        println!("{}", style("🔄 Vector Format Conversion").bold().green());
        println!("  Input: {} ({})", 
                args.input.display(), 
                detect_format(&args.input, &args.input_format)?);
        println!("  Output: {} ({})", 
                args.output.display(), 
                args.output_format);
        println!("  Output type: {}", args.output_type);
        println!();
    }
    
    // Load input vectors
    if cli.verbose {
        println!("Loading vectors from {}...", args.input.display());
    }
    
    let (mut vectors, dimension) = load_input_vectors(&args)?;
    let original_count = vectors.len();
    
    if !cli.no_progress {
        println!("✅ Loaded {} vectors of dimension {}", 
                style(original_count).bold(), style(dimension).bold());
    }
    
    // Apply skip and limit
    if args.skip > 0 {
        if args.skip >= vectors.len() {
            return Err(anyhow::anyhow!("Skip value {} >= total vectors {}", 
                                     args.skip, vectors.len()));
        }
        vectors.drain(0..args.skip);
        if !cli.no_progress {
            println!("⏭️  Skipped first {} vectors", args.skip);
        }
    }
    
    if let Some(limit) = args.limit {
        vectors.truncate(limit);
        if !cli.no_progress {
            println!("✂️  Limited to {} vectors", limit);
        }
    }
    
    // Apply preprocessing
    if args.center {
        center_vectors(&mut vectors)?;
        if !cli.no_progress {
            println!("📍 Centered vectors (subtracted mean)");
        }
    }
    
    if args.normalize {
        normalize_vectors(&mut vectors)?;
        if !cli.no_progress {
            println!("📏 Normalized vectors to unit length");
        }
    }
    
    // Show preprocessing statistics
    if args.stats {
        show_vector_statistics(&vectors, cli);
    }
    
    // Convert data type if needed
    let output_vectors = convert_data_type(&vectors, &args)?;
    
    // Save output vectors
    if cli.verbose {
        println!("Saving vectors to {}...", args.output.display());
    }
    
    save_output_vectors(&output_vectors, dimension, &args)?;
    
    if !cli.no_progress {
        println!("✅ Conversion complete!");
        println!("  Processed: {} vectors", vectors.len());
        println!("  Saved to: {}", args.output.display());
        
        if args.skip > 0 || args.limit.is_some() {
            println!("  Original count: {}", original_count);
        }
    }
    
    Ok(())
}

fn detect_format(path: &PathBuf, format_hint: &str) -> diskann::Result<String> {
    if format_hint != "auto" {
        return Ok(format_hint.to_string());
    }
    
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext.to_lowercase().as_str() {
            "fvecs" => Ok("fvecs".to_string()),
            "bvecs" => Ok("bvecs".to_string()),
            "ivecs" => Ok("ivecs".to_string()),
            "bin" => Ok("bin".to_string()),
            _ => Err(anyhow::anyhow!("Cannot auto-detect format for extension: {}", ext)),
        }
    } else {
        Err(anyhow::anyhow!("Cannot auto-detect format, no file extension"))
    }
}

fn load_input_vectors(args: &ConvertArgs) -> diskann::Result<(Vec<Vec<f32>>, usize)> {
    let format = detect_format(&args.input, &args.input_format)?;
    
    match format.as_str() {
        "fvecs" => {
            let (vectors, dim) = diskann::formats::read_fvecs(&args.input)?;
            Ok((vectors, dim))
        }
        "bvecs" => {
            let (int_vectors, dim) = diskann::formats::read_bvecs(&args.input)?;
            let vectors = int_vectors.into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            Ok((vectors, dim))
        }
        "ivecs" => {
            let (int_vectors, dim) = diskann::formats::read_ivecs(&args.input)?;
            let vectors = int_vectors.into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            Ok((vectors, dim))
        }
        "bin" => {
            let dimension = args.input_dimension
                .ok_or_else(|| anyhow::anyhow!("Dimension required for binary format"))?;
            let vectors = diskann::formats::read_binary_vectors(&args.input, dimension)?;
            Ok((vectors, dimension))
        }
        _ => Err(anyhow::anyhow!("Unsupported input format: {}", format)),
    }
}

fn center_vectors(vectors: &mut [Vec<f32>]) -> diskann::Result<()> {
    if vectors.is_empty() {
        return Ok(());
    }
    
    let dimension = vectors[0].len();
    let mut mean = vec![0.0f32; dimension];
    
    // Calculate mean
    for vector in vectors.iter() {
        for (i, &value) in vector.iter().enumerate() {
            mean[i] += value;
        }
    }
    
    for mean_val in mean.iter_mut() {
        *mean_val /= vectors.len() as f32;
    }
    
    // Subtract mean from each vector
    for vector in vectors.iter_mut() {
        for (i, value) in vector.iter_mut().enumerate() {
            *value -= mean[i];
        }
    }
    
    Ok(())
}

fn normalize_vectors(vectors: &mut [Vec<f32>]) -> diskann::Result<()> {
    for vector in vectors.iter_mut() {
        let norm: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm > 1e-8 {  // Avoid division by zero
            for value in vector.iter_mut() {
                *value /= norm;
            }
        }
    }
    
    Ok(())
}

fn show_vector_statistics(vectors: &[Vec<f32>], cli: &crate::Cli) {
    if cli.no_progress || vectors.is_empty() {
        return;
    }
    
    let dimension = vectors[0].len();
    
    // Calculate statistics across all dimensions
    let mut all_values = Vec::new();
    let mut norms = Vec::new();
    
    for vector in vectors {
        all_values.extend(vector.iter().copied());
        let norm: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        norms.push(norm);
    }
    
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let min_val = all_values[0];
    let max_val = all_values[all_values.len() - 1];
    let mean_val = all_values.iter().sum::<f32>() / all_values.len() as f32;
    let median_val = all_values[all_values.len() / 2];
    
    let min_norm = norms[0];
    let max_norm = norms[norms.len() - 1];
    let mean_norm = norms.iter().sum::<f32>() / norms.len() as f32;
    
    println!("\n📊 Vector Statistics:");
    println!("  Vectors: {}", vectors.len());
    println!("  Dimension: {}", dimension);
    println!("  Value range: [{:.6}, {:.6}]", min_val, max_val);
    println!("  Value mean: {:.6}", mean_val);
    println!("  Value median: {:.6}", median_val);
    println!("  Norm range: [{:.6}, {:.6}]", min_norm, max_norm);
    println!("  Norm mean: {:.6}", mean_norm);
}

fn convert_data_type(vectors: &[Vec<f32>], args: &ConvertArgs) -> diskann::Result<Vec<u8>> {
    match args.output_type.to_lowercase().as_str() {
        "float32" | "f32" => {
            // No conversion needed, just serialize as bytes
            let mut result = Vec::new();
            for vector in vectors {
                for &value in vector {
                    result.extend_from_slice(&value.to_le_bytes());
                }
            }
            Ok(result)
        }
        "float16" | "f16" => {
            // Convert to float16
            let mut result = Vec::new();
            for vector in vectors {
                for &value in vector {
                    let f16_val = half::f16::from_f32(value);
                    result.extend_from_slice(&f16_val.to_le_bytes());
                }
            }
            Ok(result)
        }
        "int8" | "i8" => {
            // Quantize to int8
            let params = calculate_quantization_params(vectors, args)?;
            let mut result = Vec::new();
            for vector in vectors {
                for &value in vector {
                    let quantized = quantize_to_int8(value, &params);
                    result.push(quantized as u8);
                }
            }
            Ok(result)
        }
        "uint8" | "u8" => {
            // Quantize to uint8
            let params = calculate_quantization_params(vectors, args)?;
            let mut result = Vec::new();
            for vector in vectors {
                for &value in vector {
                    let quantized = quantize_to_uint8(value, &params);
                    result.push(quantized);
                }
            }
            Ok(result)
        }
        _ => Err(anyhow::anyhow!("Unsupported output type: {}", args.output_type)),
    }
}

fn calculate_quantization_params(vectors: &[Vec<f32>], args: &ConvertArgs) -> diskann::Result<QuantizationParams> {
    let mut all_values: Vec<f32> = vectors.iter().flatten().cloned().collect();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    match args.quantization.as_str() {
        "minmax" => {
            Ok(QuantizationParams {
                min_val: all_values[0],
                max_val: all_values[all_values.len() - 1],
                mean: 0.0,
                std: 1.0,
            })
        }
        "standard" => {
            let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
            let variance = all_values.iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<f32>() / all_values.len() as f32;
            let std = variance.sqrt();
            
            Ok(QuantizationParams {
                min_val: mean - 3.0 * std,
                max_val: mean + 3.0 * std,
                mean,
                std,
            })
        }
        _ => Err(anyhow::anyhow!("Unsupported quantization method: {}", args.quantization)),
    }
}

fn quantize_to_int8(value: f32, params: &QuantizationParams) -> i8 {
    let normalized = (value - params.min_val) / (params.max_val - params.min_val);
    let scaled = normalized * 255.0 - 128.0;
    scaled.clamp(-128.0, 127.0) as i8
}

fn quantize_to_uint8(value: f32, params: &QuantizationParams) -> u8 {
    let normalized = (value - params.min_val) / (params.max_val - params.min_val);
    let scaled = normalized * 255.0;
    scaled.clamp(0.0, 255.0) as u8
}

fn save_output_vectors(data: &[u8], dimension: usize, args: &ConvertArgs) -> diskann::Result<()> {
    match args.output_format.to_lowercase().as_str() {
        "fvecs" => {
            // Convert bytes back to f32 vectors for fvecs format
            if args.output_type != "float32" {
                return Err(anyhow::anyhow!("fvecs format requires float32 output type"));
            }
            
            let num_vectors = data.len() / (dimension * 4);
            let mut vectors = Vec::with_capacity(num_vectors);
            
            for i in 0..num_vectors {
                let mut vector = Vec::with_capacity(dimension);
                for j in 0..dimension {
                    let offset = (i * dimension + j) * 4;
                    let bytes = &data[offset..offset + 4];
                    let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    vector.push(value);
                }
                vectors.push(vector);
            }
            
            diskann::formats::write_fvecs(&args.output, &vectors)?;
        }
        "bvecs" => {
            // For bvecs, data should be uint8
            let num_vectors = data.len() / dimension;
            let mut vectors = Vec::with_capacity(num_vectors);
            
            for i in 0..num_vectors {
                let start = i * dimension;
                let end = start + dimension;
                vectors.push(data[start..end].to_vec());
            }
            
            diskann::formats::write_bvecs(&args.output, &vectors)?;
        }
        "bin" => {
            // Write raw binary data
            use std::fs::File;
            use std::io::Write;
            
            let mut file = File::create(&args.output)?;
            file.write_all(data)?;
        }
        _ => {
            return Err(anyhow::anyhow!("Unsupported output format: {}", args.output_format));
        }
    }
    
    Ok(())
}