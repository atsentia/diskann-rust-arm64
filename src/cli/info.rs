//! Information and analysis command
//!
//! This module provides utilities for analyzing vector files and indices,
//! showing statistics, and validating data integrity.

use clap::Args;
use console::style;
use std::path::PathBuf;
use std::collections::HashMap;

use crate::*;

#[derive(Args)]
pub struct InfoArgs {
    /// Path to file to analyze
    #[arg(short, long)]
    pub input: PathBuf,
    
    /// File format (auto, fvecs, bvecs, ivecs, bin, index)
    #[arg(long, default_value = "auto")]
    pub format: String,
    
    /// Dimension for binary format
    #[arg(long)]
    pub dimension: Option<usize>,
    
    /// Show detailed statistics
    #[arg(long)]
    pub detailed: bool,
    
    /// Show first N vectors
    #[arg(long)]
    pub preview: Option<usize>,
    
    /// Analyze vector distribution
    #[arg(long)]
    pub distribution: bool,
    
    /// Check for duplicate vectors
    #[arg(long)]
    pub duplicates: bool,
    
    /// Validate data integrity
    #[arg(long)]
    pub validate: bool,
    
    /// Show memory usage estimation
    #[arg(long)]
    pub memory: bool,
}

pub fn run(args: InfoArgs, cli: &crate::Cli) -> crate::Result<()> {
    if !cli.no_progress {
        println!("{}", style("ℹ️  File Analysis").bold().blue());
        println!("  File: {}", args.input.display());
        println!();
    }
    
    let format = detect_format(&args.input, &args.format)?;
    
    match format.as_str() {
        "fvecs" | "bvecs" | "ivecs" | "bin" => {
            analyze_vectors(&args, &format, cli)?;
        }
        "index" => {
            analyze_index(&args, cli)?;
        }
        _ => {
            return Err(anyhow::anyhow!("Unsupported format: {}", format));
        }
    }
    
    Ok(())
}

fn detect_format(path: &PathBuf, format_hint: &str) -> crate::Result<String> {
    if format_hint != "auto" {
        return Ok(format_hint.to_string());
    }
    
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext.to_lowercase().as_str() {
            "fvecs" => Ok("fvecs".to_string()),
            "bvecs" => Ok("bvecs".to_string()), 
            "ivecs" => Ok("ivecs".to_string()),
            "bin" => Ok("bin".to_string()),
            "index" | "diskann" => Ok("index".to_string()),
            _ => Err(anyhow::anyhow!("Cannot auto-detect format for extension: {}", ext)),
        }
    } else {
        Err(anyhow::anyhow!("Cannot auto-detect format, no file extension"))
    }
}

fn analyze_vectors(args: &InfoArgs, format: &str, cli: &crate::Cli) -> crate::Result<()> {
    // Load vectors
    let (vectors, dimension) = load_vectors(args, format)?;
    
    if !cli.no_progress {
        println!("📊 Basic Information:");
        println!("  Format: {}", style(format).bold());
        println!("  Vectors: {}", style(vectors.len()).bold().green());
        println!("  Dimension: {}", style(dimension).bold().green());
        println!("  File size: {}", format_file_size(&args.input)?);
    }
    
    // Show preview if requested
    if let Some(preview_count) = args.preview {
        show_vector_preview(&vectors, preview_count, cli);
    }
    
    // Show detailed statistics
    if args.detailed {
        show_detailed_statistics(&vectors, cli);
    }
    
    // Analyze distribution
    if args.distribution {
        analyze_distribution(&vectors, cli);
    }
    
    // Check for duplicates
    if args.duplicates {
        check_duplicates(&vectors, cli);
    }
    
    // Validate data
    if args.validate {
        validate_vectors(&vectors, cli);
    }
    
    // Show memory usage
    if args.memory {
        show_memory_usage(&vectors, dimension, cli);
    }
    
    Ok(())
}

fn load_vectors(args: &InfoArgs, format: &str) -> crate::Result<(Vec<Vec<f32>>, usize)> {
    match format {
        "fvecs" => {
            let (vectors, dim) = crate::formats::read_fvecs(&args.input)?;
            Ok((vectors, dim))
        }
        "bvecs" => {
            let (int_vectors, dim) = crate::formats::read_bvecs(&args.input)?;
            let vectors = int_vectors.into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            Ok((vectors, dim))
        }
        "ivecs" => {
            let (int_vectors, dim) = crate::formats::read_ivecs(&args.input)?;
            let vectors = int_vectors.into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            Ok((vectors, dim))
        }
        "bin" => {
            let dimension = args.dimension
                .ok_or_else(|| anyhow::anyhow!("Dimension required for binary format"))?;
            let vectors = crate::formats::read_binary_vectors(&args.input, dimension)?;
            Ok((vectors, dimension))
        }
        _ => Err(anyhow::anyhow!("Unsupported format: {}", format)),
    }
}

fn format_file_size(path: &PathBuf) -> crate::Result<String> {
    let metadata = std::fs::metadata(path)?;
    let size = metadata.len();
    
    if size < 1024 {
        Ok(format!("{} B", size))
    } else if size < 1024 * 1024 {
        Ok(format!("{:.1} KB", size as f64 / 1024.0))
    } else if size < 1024 * 1024 * 1024 {
        Ok(format!("{:.1} MB", size as f64 / (1024.0 * 1024.0)))
    } else {
        Ok(format!("{:.1} GB", size as f64 / (1024.0 * 1024.0 * 1024.0)))
    }
}

fn show_vector_preview(vectors: &[Vec<f32>], count: usize, cli: &crate::Cli) {
    if cli.no_progress {
        return;
    }
    
    println!("\n👁️  Vector Preview (first {} vectors):", count.min(vectors.len()));
    
    for (i, vector) in vectors.iter().take(count).enumerate() {
        print!("  [{:3}]: [", i);
        
        // Show first few dimensions
        let preview_dims = 8.min(vector.len());
        for (j, &value) in vector.iter().take(preview_dims).enumerate() {
            if j > 0 { print!(", "); }
            print!("{:8.4}", value);
        }
        
        if vector.len() > preview_dims {
            print!(", ... ({} more)", vector.len() - preview_dims);
        }
        
        println!("]");
    }
}

fn show_detailed_statistics(vectors: &[Vec<f32>], cli: &crate::Cli) {
    if cli.no_progress || vectors.is_empty() {
        return;
    }
    
    println!("\n📈 Detailed Statistics:");
    
    // Collect all values
    let mut all_values: Vec<f32> = vectors.iter().flatten().cloned().collect();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Calculate statistics
    let min_val = all_values[0];
    let max_val = all_values[all_values.len() - 1];
    let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
    let median = all_values[all_values.len() / 2];
    
    let variance = all_values.iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f32>() / all_values.len() as f32;
    let std_dev = variance.sqrt();
    
    // Calculate percentiles
    let p25 = all_values[all_values.len() / 4];
    let p75 = all_values[all_values.len() * 3 / 4];
    let p95 = all_values[all_values.len() * 95 / 100];
    let p99 = all_values[all_values.len() * 99 / 100];
    
    println!("  Range: [{:.6}, {:.6}]", min_val, max_val);
    println!("  Mean: {:.6}", mean);
    println!("  Median: {:.6}", median);
    println!("  Std Dev: {:.6}", std_dev);
    println!("  Percentiles:");
    println!("    25th: {:.6}", p25);
    println!("    75th: {:.6}", p75);
    println!("    95th: {:.6}", p95);
    println!("    99th: {:.6}", p99);
    
    // Calculate norms
    let mut norms: Vec<f32> = vectors.iter()
        .map(|v| v.iter().map(|&x| x * x).sum::<f32>().sqrt())
        .collect();
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mean_norm = norms.iter().sum::<f32>() / norms.len() as f32;
    let min_norm = norms[0];
    let max_norm = norms[norms.len() - 1];
    let median_norm = norms[norms.len() / 2];
    
    println!("  Vector Norms:");
    println!("    Range: [{:.6}, {:.6}]", min_norm, max_norm);
    println!("    Mean: {:.6}", mean_norm);
    println!("    Median: {:.6}", median_norm);
}

fn analyze_distribution(vectors: &[Vec<f32>], cli: &crate::Cli) {
    if cli.no_progress || vectors.is_empty() {
        return;
    }
    
    println!("\n📊 Value Distribution Analysis:");
    
    // Create histogram
    let all_values: Vec<f32> = vectors.iter().flatten().cloned().collect();
    let min_val = all_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = all_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    let num_bins = 20;
    let bin_width = (max_val - min_val) / num_bins as f32;
    let mut histogram = vec![0usize; num_bins];
    
    for &value in &all_values {
        let bin = ((value - min_val) / bin_width) as usize;
        let bin = bin.min(num_bins - 1);
        histogram[bin] += 1;
    }
    
    // Find max count for scaling
    let max_count = *histogram.iter().max().unwrap_or(&0);
    let scale = 50.0 / max_count as f32;
    
    println!("  Range: [{:.3}, {:.3}]", min_val, max_val);
    println!("  Histogram:");
    
    for (i, &count) in histogram.iter().enumerate() {
        let bin_start = min_val + i as f32 * bin_width;
        let bin_end = bin_start + bin_width;
        let bar_length = (count as f32 * scale) as usize;
        let bar = "█".repeat(bar_length);
        
        println!("    [{:8.3}, {:8.3}): {:6} {}", 
                bin_start, bin_end, count, bar);
    }
}

fn check_duplicates(vectors: &[Vec<f32>], cli: &crate::Cli) {
    if cli.no_progress || vectors.is_empty() {
        return;
    }
    
    println!("\n🔍 Duplicate Analysis:");
    
    let mut vector_map: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
    
    // Hash vectors (convert to fixed precision to handle floating point issues) 
    for (i, vector) in vectors.iter().enumerate() {
        let hash_vector: Vec<u32> = vector.iter()
            .map(|&x| (x * 1000000.0) as u32) // 6 decimal places precision
            .collect();
        
        vector_map.entry(hash_vector).or_insert_with(Vec::new).push(i);
    }
    
    let mut duplicate_groups = 0;
    let mut total_duplicates = 0;
    
    for (_, indices) in vector_map.iter() {
        if indices.len() > 1 {
            duplicate_groups += 1;
            total_duplicates += indices.len() - 1; // Don't count the original
        }
    }
    
    println!("  Unique vectors: {}", vector_map.len());
    println!("  Duplicate groups: {}", duplicate_groups);
    println!("  Total duplicates: {}", total_duplicates);
    
    if duplicate_groups > 0 && duplicate_groups <= 5 {
        println!("  Example duplicate groups:");
        let mut shown = 0;
        for (_, indices) in vector_map.iter() {
            if indices.len() > 1 && shown < 3 {
                println!("    Group: {:?}", indices);
                shown += 1;
            }
        }
    }
}

fn validate_vectors(vectors: &[Vec<f32>], cli: &crate::Cli) {
    if cli.no_progress {
        return;
    }
    
    println!("\n✅ Data Validation:");
    
    let mut issues = Vec::new();
    
    // Check for consistent dimensions
    if !vectors.is_empty() {
        let expected_dim = vectors[0].len();
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != expected_dim {
                issues.push(format!("Vector {} has dimension {}, expected {}", 
                                  i, vector.len(), expected_dim));
                if issues.len() >= 5 { break; } // Limit error messages
            }
        }
    }
    
    // Check for invalid values
    let mut nan_count = 0;
    let mut inf_count = 0;
    
    for (i, vector) in vectors.iter().enumerate() {
        for (j, &value) in vector.iter().enumerate() {
            if value.is_nan() {
                nan_count += 1;
                if nan_count <= 5 {
                    issues.push(format!("NaN found at vector {}[{}]", i, j));
                }
            } else if value.is_infinite() {
                inf_count += 1;
                if inf_count <= 5 {
                    issues.push(format!("Infinite value found at vector {}[{}]", i, j));
                }
            }
        }
    }
    
    if issues.is_empty() {
        println!("  ✅ All vectors are valid");
        println!("  ✅ Consistent dimensions");
        println!("  ✅ No NaN or infinite values");
    } else {
        println!("  ⚠️  Found {} issues:", issues.len());
        for issue in issues.iter().take(10) {
            println!("    • {}", issue);
        }
        if issues.len() > 10 {
            println!("    ... and {} more", issues.len() - 10);
        }
    }
    
    if nan_count > 5 {
        println!("  ⚠️  Total NaN values: {}", nan_count);
    }
    if inf_count > 5 {
        println!("  ⚠️  Total infinite values: {}", inf_count);
    }
}

fn show_memory_usage(vectors: &[Vec<f32>], dimension: usize, cli: &crate::Cli) {
    if cli.no_progress {
        return;
    }
    
    println!("\n💾 Memory Usage Estimation:");
    
    let vectors_size = vectors.len() * dimension * std::mem::size_of::<f32>();
    let metadata_size = vectors.len() * std::mem::size_of::<usize>(); // Approximate
    let total_size = vectors_size + metadata_size;
    
    println!("  Raw vector data: {} MB", vectors_size / (1024 * 1024));
    println!("  Metadata overhead: {} KB", metadata_size / 1024);
    println!("  Total memory: {} MB", total_size / (1024 * 1024));
    println!("  Per vector: {} bytes", total_size / vectors.len());
    
    // Estimate index memory usage
    let graph_edges_per_node = 64; // Typical max degree
    let graph_size = vectors.len() * graph_edges_per_node * std::mem::size_of::<usize>();
    
    println!("  Estimated Vamana graph: {} MB", graph_size / (1024 * 1024));
    println!("  Total with index: {} MB", (total_size + graph_size) / (1024 * 1024));
    
    // PQ compression estimates
    let compression_ratios = [4, 8, 16, 32, 64];
    println!("  PQ Compression estimates:");
    for ratio in compression_ratios {
        let compressed_size = vectors_size / ratio;
        println!("    {}x compression: {} MB", ratio, compressed_size / (1024 * 1024));
    }
}

fn analyze_index(args: &InfoArgs, cli: &crate::Cli) -> crate::Result<()> {
    if !cli.no_progress {
        println!("📋 Index Analysis:");
        println!("  File: {}", args.input.display());
    }
    
    // TODO: Implement index analysis
    println!("  ⚠️  Index analysis not yet implemented");
    println!("  This would show:");
    println!("    • Index type (standard, PQ, etc.)");
    println!("    • Graph statistics (nodes, edges, degree distribution)");
    println!("    • Distance metric and parameters");
    println!("    • Memory usage breakdown");
    println!("    • Build timestamp and metadata");
    
    Ok(())
}