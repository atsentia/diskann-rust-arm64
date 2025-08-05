//! Index searching command
//!
//! This module provides functionality for searching existing DiskANN indices
//! with various query types and output formats.

use clap::Args;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;

use crate::*;

#[derive(Args)]
pub struct SearchArgs {
    /// Path to index file
    #[arg(short, long)]
    pub index: PathBuf,
    
    /// Path to query vectors file
    #[arg(short, long)]
    pub queries: PathBuf,
    
    /// Number of nearest neighbors to find
    #[arg(short, long, default_value = "10")]
    pub k: usize,
    
    /// Search list size (higher = better recall, slower)
    #[arg(short, long, default_value = "50")]
    pub search_list_size: usize,
    
    /// Output results file (optional)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    
    /// Query file format (auto, fvecs, bvecs, ivecs, bin)
    #[arg(long, default_value = "auto")]
    pub format: String,
    
    /// Dimension for binary format queries
    #[arg(long)]
    pub dimension: Option<usize>,
    
    /// Show distances in output
    #[arg(long)]
    pub show_distances: bool,
    
    /// Use accurate search (asymmetric PQ distance)
    #[arg(long)]
    pub accurate: bool,
    
    /// Filter by labels (comma-separated)
    #[arg(long)]
    pub filter_labels: Option<String>,
    
    /// Range search: find all within distance
    #[arg(long)]
    pub range: Option<f32>,
    
    /// Maximum number of queries to process
    #[arg(long)]
    pub max_queries: Option<usize>,
}

pub fn run(args: SearchArgs, cli: &crate::Cli) -> diskann::Result<()> {
    let start_time = Instant::now();
    
    if !cli.no_progress {
        println!("{}", style("🔍 Searching DiskANN Index").bold().green());
        println!("  Index: {}", args.index.display());
        println!("  Queries: {}", args.queries.display());
        println!("  k: {}", args.k);
        println!("  Search list size: {}", args.search_list_size);
        println!();
    }
    
    // Load index
    if cli.verbose {
        println!("Loading index from {}...", args.index.display());
    }
    
    let index = crate::index::memory::MemoryIndex::load(&args.index)?;
    
    if !cli.no_progress {
        println!("✅ Loaded index with {} vectors", 
                style(index.size()).bold());
    }
    
    // Load queries
    if cli.verbose {
        println!("Loading queries from {}...", args.queries.display());
    }
    
    let (queries, query_dim) = load_queries(&args)?;
    let num_queries = queries.len().min(args.max_queries.unwrap_or(usize::MAX));
    
    if !cli.no_progress {
        println!("✅ Loaded {} queries of dimension {}", 
                style(num_queries).bold(), style(query_dim).bold());
    }
    
    // Execute searches
    let pb: Option<ProgressBar> = if !cli.no_progress {
        let pb = ProgressBar::new(num_queries as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} queries ({per_sec}) {eta}")
            .unwrap()
            .progress_chars("#>-"));
        Some(pb)
    } else {
        None
    };
    
    let mut all_results = Vec::new();
    let mut total_search_time = 0.0f64;
    
    for (query_idx, query) in queries.iter().take(num_queries).enumerate() {
        let search_start = Instant::now();
        
        // Execute search based on type
        let results = if let Some(range_dist) = args.range {
            // Range search
            execute_range_search(query, range_dist, &index, &args)?
        } else if args.filter_labels.is_some() {
            // Filtered search
            execute_filtered_search(query, &index, &args)?
        } else {
            // Standard k-NN search
            execute_standard_search(query, &index, &args)?
        };
        
        let search_time = search_start.elapsed();
        total_search_time += search_time.as_secs_f64();
        
        all_results.push((query_idx, results));
        
        if let Some(ref pb) = pb {
            pb.inc(1);
        }
        
        // Show progress for first few queries if verbose
        if cli.verbose && query_idx < 3 {
            println!("Query {}: {} results in {:.2}ms", 
                    query_idx, 
                    all_results[query_idx].1.len(),
                    search_time.as_secs_f64() * 1000.0);
        }
    }
    
    if let Some(pb) = pb {
        pb.finish_with_message("Search complete");
    }
    
    // Calculate and display statistics
    let avg_search_time_ms = (total_search_time / num_queries as f64) * 1000.0;
    let qps = num_queries as f64 / total_search_time;
    
    if !cli.no_progress {
        println!("\n📊 Search Statistics:");
        println!("  Queries processed: {}", style(num_queries).bold());
        println!("  Average search time: {:.2} ms", style(format!("{:.2}", avg_search_time_ms)).bold());
        println!("  Queries per second: {:.0} QPS", style(format!("{:.0}", qps)).bold());
        println!("  Total time: {}", style(humantime::format_duration(start_time.elapsed())).bold());
    }
    
    // Output results
    if let Some(ref output_path) = args.output {
        write_results(&all_results, output_path, &args)?;
        if !cli.no_progress {
            println!("📁 Results saved to: {}", output_path.display());
        }
    } else {
        // Display first few results
        display_sample_results(&all_results, &args, cli)?;
    }
    
    Ok(())
}

fn load_queries(args: &SearchArgs) -> diskann::Result<(Vec<Vec<f32>>, usize)> {
    let format = detect_format(&args.queries, &args.format)?;
    
    match format.as_str() {
        "fvecs" => {
            let (vectors, dim) = crate::formats::read_fvecs(&args.queries)?;
            Ok((vectors, dim))
        }
        "bvecs" => {
            let (int_vectors, dim) = crate::formats::read_bvecs(&args.queries)?;
            let vectors = int_vectors.into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            Ok((vectors, dim))
        }
        "ivecs" => {
            let (int_vectors, dim) = crate::formats::read_ivecs(&args.queries)?;
            let vectors = int_vectors.into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            Ok((vectors, dim))
        }
        "bin" => {
            let dimension = args.dimension
                .ok_or_else(|| anyhow::anyhow!("Dimension required for binary format"))?;
            let vectors = crate::formats::read_binary_vectors(&args.queries, dimension)?;
            Ok((vectors, dimension))
        }
        _ => Err(anyhow::anyhow!("Unsupported format: {}", format)),
    }
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

fn execute_standard_search(
    query: &[f32],
    index: &crate::index::memory::MemoryIndex,
    args: &SearchArgs,
) -> crate::Result<Vec<(usize, f32)>> {
    use crate::Index;
    index.search(query, args.k)
}

fn execute_range_search(
    _query: &[f32],
    _range_dist: f32,
    _index: &crate::index::memory::MemoryIndex,
    _args: &SearchArgs,
) -> crate::Result<Vec<(usize, f32)>> {
    // TODO: Implement range search
    Ok(vec![(0, 0.3), (5, 0.45)]) // Dummy results
}

fn execute_filtered_search(
    _query: &[f32],
    _index: &crate::index::memory::MemoryIndex,
    _args: &SearchArgs,
) -> crate::Result<Vec<(usize, f32)>> {
    // TODO: Implement filtered search with labels
    Ok(vec![(2, 0.6), (7, 0.8)]) // Dummy results
}

fn write_results(
    results: &[(usize, Vec<(usize, f32)>)],
    output_path: &PathBuf,
    args: &SearchArgs,
) -> diskann::Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};
    
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    
    for (query_idx, neighbors) in results {
        write!(writer, "{}", query_idx)?;
        
        for (neighbor_id, distance) in neighbors {
            if args.show_distances {
                write!(writer, " {}:{:.6}", neighbor_id, distance)?;
            } else {
                write!(writer, " {}", neighbor_id)?;
            }
        }
        
        writeln!(writer)?;
    }
    
    writer.flush()?;
    Ok(())
}

fn display_sample_results(
    results: &[(usize, Vec<(usize, f32)>)],
    args: &SearchArgs,
    cli: &crate::Cli,
) -> diskann::Result<()> {
    if cli.no_progress {
        return Ok(());
    }
    
    println!("\n🎯 Sample Results (first 3 queries):");
    
    for (query_idx, neighbors) in results.iter().take(3) {
        println!("\nQuery {}:", style(query_idx).bold().cyan());
        
        for (i, (neighbor_id, distance)) in neighbors.iter().take(args.k.min(5)).enumerate() {
            if args.show_distances {
                println!("  {}. ID: {} (distance: {:.6})", 
                        i + 1, 
                        style(neighbor_id).bold(),
                        style(format!("{:.6}", distance)).dim());
            } else {
                println!("  {}. ID: {}", i + 1, style(neighbor_id).bold());
            }
        }
        
        if neighbors.len() > 5 {
            println!("  ... and {} more", neighbors.len() - 5);
        }
    }
    
    if results.len() > 3 {
        println!("\n... and {} more queries", results.len() - 3);
    }
    
    Ok(())
}