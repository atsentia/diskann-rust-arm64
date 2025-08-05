//! Index building command
//!
//! This module provides functionality for building DiskANN indices from various
//! vector formats with configurable parameters.

use clap::Args;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;

use crate::*;
use crate::pq::{PQParams, PQIndex};

#[derive(Args)]
pub struct BuildArgs {
    /// Path to input vectors file
    #[arg(short, long)]
    pub input: PathBuf,
    
    /// Path to output index file
    #[arg(short, long)]
    pub output: PathBuf,
    
    /// Vector dimension (auto-detected for supported formats)
    #[arg(short, long)]
    pub dimension: Option<usize>,
    
    /// Distance metric (l2, cosine, inner_product)
    #[arg(short = 'm', long, default_value = "l2")]
    pub metric: String,
    
    /// Maximum degree in graph
    #[arg(long, default_value = "64")]
    pub max_degree: usize,
    
    /// Search list size during construction
    #[arg(long, default_value = "100")]
    pub search_list_size: usize,
    
    /// Alpha parameter for RobustPrune
    #[arg(long, default_value = "1.2")]
    pub alpha: f32,
    
    /// Input file format (auto, fvecs, bvecs, ivecs, bin)
    #[arg(long, default_value = "auto")]
    pub format: String,
    
    /// Enable Product Quantization
    #[arg(long)]
    pub use_pq: bool,
    
    /// Number of PQ subspaces
    #[arg(long, default_value = "8")]
    pub pq_subspaces: usize,
    
    /// Bits per PQ subquantizer
    #[arg(long, default_value = "8")]
    pub pq_bits: usize,
    
    /// Labels file (optional, one label per line)
    #[arg(long)]
    pub labels: Option<PathBuf>,
    
    /// Number of threads to use (0 = auto)
    #[arg(short = 'j', long, default_value = "0")]
    pub threads: usize,
}

pub fn run(args: BuildArgs, cli: &crate::Cli) -> crate::Result<()> {
    let start_time = Instant::now();
    
    // Parse distance metric
    let distance = match args.metric.to_lowercase().as_str() {
        "l2" | "euclidean" => Distance::L2,
        "cosine" => Distance::Cosine,
        "inner_product" | "ip" => Distance::InnerProduct,
        _ => return Err(anyhow::anyhow!("Invalid distance metric: {}", args.metric)),
    };
    
    if !cli.no_progress {
        println!("{}", style("📊 Building DiskANN Index").bold().green());
        println!("  Input: {}", args.input.display());
        println!("  Output: {}", args.output.display());
        println!("  Metric: {:?}", distance);
        println!("  Max degree: {}", args.max_degree);
        println!("  PQ enabled: {}", args.use_pq);
        println!();
    }
    
    // Load vectors
    if cli.verbose {
        println!("Loading vectors from {}...", args.input.display());
    }
    
    let (vectors, dimension) = load_vectors(&args)?;
    let num_vectors = vectors.len();
    
    if !cli.no_progress {
        println!("✅ Loaded {} vectors of dimension {}", 
                style(num_vectors).bold(), style(dimension).bold());
    }
    
    // Load labels if provided
    let labels = if let Some(ref labels_path) = args.labels {
        if cli.verbose {
            println!("Loading labels from {}...", labels_path.display());
        }
        Some(load_labels(labels_path, num_vectors)?)
    } else {
        None
    };
    
    // Build index based on configuration
    if args.use_pq {
        build_pq_index(vectors, dimension, distance, &args, labels, cli)?;
    } else {
        build_standard_index(vectors, dimension, distance, &args, labels, cli)?;
    }
    
    let elapsed = start_time.elapsed();
    if !cli.no_progress {
        println!("\n🎉 Index built successfully in {}", 
                style(humantime::format_duration(elapsed)).bold().green());
        println!("📁 Saved to: {}", args.output.display());
    }
    
    Ok(())
}

fn load_vectors(args: &BuildArgs) -> crate::Result<(Vec<Vec<f32>>, usize)> {
    let format = detect_format(&args.input, &args.format)?;
    
    match format.as_str() {
        "fvecs" => {
            let (vectors, dim) = crate::formats::read_fvecs(&args.input)?;
            Ok((vectors, dim))
        }
        "bvecs" => {
            let (int_vectors, dim) = crate::formats::read_bvecs(&args.input)?;
            // Convert to float vectors
            let vectors = int_vectors.into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            Ok((vectors, dim))
        }
        "ivecs" => {
            let (int_vectors, dim) = crate::formats::read_ivecs(&args.input)?;
            // Convert to float vectors
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

fn detect_format(path: &PathBuf, format_hint: &str) -> crate::Result<String> {
    if format_hint != "auto" {
        return Ok(format_hint.to_string());
    }
    
    // Auto-detect from file extension
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

fn load_labels(path: &PathBuf, expected_count: usize) -> crate::Result<Vec<Vec<u32>>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut labels = Vec::new();
    
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        
        if line.is_empty() {
            labels.push(vec![]);
            continue;
        }
        
        // Parse comma-separated labels
        let mut parsed_labels = Vec::new();
        for label_str in line.split(',') {
            match label_str.trim().parse::<u32>() {
                Ok(label) => parsed_labels.push(label),
                Err(e) => return Err(anyhow::anyhow!(
                    "Failed to parse label '{}' at line {}: {}", label_str.trim(), line_num + 1, e
                )),
            }
        }
        labels.push(parsed_labels);
    }
    
    if labels.len() != expected_count {
        return Err(anyhow::anyhow!(
            "Label count mismatch: expected {}, got {}", expected_count, labels.len()
        ));
    }
    
    Ok(labels)
}

fn build_standard_index(
    vectors: Vec<Vec<f32>>,
    dimension: usize,
    distance: Distance,
    args: &BuildArgs,
    _labels: Option<Vec<Vec<u32>>>,
    cli: &crate::Cli,
) -> crate::Result<()> {
    if !cli.no_progress {
        println!("🏗️  Building standard Vamana index...");
    }
    
    let pb = if !cli.no_progress {
        let pb = ProgressBar::new(vectors.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        Some(pb)
    } else {
        None
    };
    
    // Create index builder
    let index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(distance)
        .max_degree(args.max_degree)
        .search_list_size(args.search_list_size)
        .alpha(args.alpha)
        .build_memory_index(vectors)?;
    
    if let Some(pb) = pb {
        pb.finish_with_message("Index built");
    }
    
    // Save index
    if cli.verbose {
        println!("Saving index to {}...", args.output.display());
    }
    
    index.save(&args.output)?;
    
    if !cli.no_progress {
        println!("✅ Index saved successfully");
    }
    
    Ok(())
}

fn build_pq_index(
    vectors: Vec<Vec<f32>>,
    dimension: usize,
    distance: Distance,
    args: &BuildArgs,
    _labels: Option<Vec<Vec<u32>>>,
    cli: &crate::Cli,
) -> crate::Result<()> {
    if !cli.no_progress {
        println!("🗜️  Building PQ-compressed index...");
        println!("  Subspaces: {}", args.pq_subspaces);
        println!("  Bits per subquantizer: {}", args.pq_bits);
    }
    
    // Validate PQ parameters
    if dimension % args.pq_subspaces != 0 {
        return Err(anyhow::anyhow!(
            "Dimension {} must be divisible by number of subspaces {}",
            dimension, args.pq_subspaces
        ));
    }
    
    let pb: Option<ProgressBar> = if !cli.no_progress {
        let pb = ProgressBar::new(100);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}% {msg}")
            .unwrap()
            .progress_chars("#>-"));
        Some(pb)
    } else {
        None
    };
    
    // Create PQ parameters
    let pq_params = PQParams::new(args.pq_subspaces, args.pq_bits);
    
    // Build PQ index
    let mut pq_index = PQIndex::new(pq_params, dimension, distance)?;
    
    if let Some(ref pb) = pb {
        pb.set_message("Training quantizer...");
        pb.set_position(10);
    }
    
    // Build the index (this trains PQ and builds graph)
    pq_index.build(vectors)?;
    
    if let Some(ref pb) = pb {
        pb.set_message("Index complete");
        pb.finish();
    }
    
    // Show compression statistics
    let stats = pq_index.memory_stats();
    if !cli.no_progress {
        println!("\n📊 PQ Compression Statistics:");
        println!("  Vectors: {}", stats.num_vectors);
        println!("  Original size: {} MB", stats.original_size_bytes / (1024 * 1024));
        println!("  Compressed size: {} MB", stats.compressed_size_bytes / (1024 * 1024));
        println!("  Compression ratio: {:.1}x", stats.compression_ratio);
        println!("  Total memory: {} MB", stats.total_memory_bytes / (1024 * 1024));
    }
    
    // TODO: Implement PQ index serialization
    println!("⚠️  PQ index serialization not yet implemented");
    
    Ok(())
}