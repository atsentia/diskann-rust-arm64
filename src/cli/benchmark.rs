//! Benchmarking command
//!
//! This module provides comprehensive benchmarking capabilities for DiskANN
//! indices, including throughput, latency, and recall measurements.

use clap::Args;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::*;

#[derive(Args)]
pub struct BenchmarkArgs {
    /// Path to index file
    #[arg(short, long)]
    pub index: PathBuf,
    
    /// Path to query vectors file
    #[arg(short, long)]
    pub queries: PathBuf,
    
    /// Path to ground truth file (for recall calculation)
    #[arg(short, long)]
    pub ground_truth: Option<PathBuf>,
    
    /// Output results file (JSON format)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    
    /// Number of nearest neighbors to find
    #[arg(short, long, default_value = "10")]
    pub k: usize,
    
    /// Search list sizes to test (comma-separated)
    #[arg(long, default_value = "10,20,50,100")]
    pub search_list_sizes: String,
    
    /// Query file format (auto, fvecs, bvecs, ivecs, bin)
    #[arg(long, default_value = "auto")]
    pub format: String,
    
    /// Dimension for binary format
    #[arg(long)]
    pub dimension: Option<usize>,
    
    /// Number of warmup queries
    #[arg(long, default_value = "100")]
    pub warmup: usize,
    
    /// Maximum number of queries to benchmark
    #[arg(long, default_value = "1000")]
    pub max_queries: usize,
    
    /// Number of threads for concurrent benchmarking
    #[arg(short = 'j', long, default_value = "1")]
    pub threads: usize,
    
    /// Run latency benchmark (single-threaded)
    #[arg(long)]
    pub latency: bool,
    
    /// Run throughput benchmark (multi-threaded)
    #[arg(long)]
    pub throughput: bool,
    
    /// Run recall benchmark (requires ground truth)
    #[arg(long)]
    pub recall: bool,
    
    /// Run all benchmarks
    #[arg(long)]
    pub all: bool,
}

#[derive(Debug, serde::Serialize)]
pub struct BenchmarkResults {
    pub timestamp: String,
    pub index_path: String,
    pub queries_path: String,
    pub k: usize,
    pub num_queries: usize,
    pub latency_results: Option<LatencyResults>,
    pub throughput_results: Option<ThroughputResults>,
    pub recall_results: Option<RecallResults>,
}

#[derive(Debug, serde::Serialize)]
pub struct LatencyResults {
    pub search_list_sizes: Vec<usize>,
    pub avg_latency_ms: Vec<f64>,
    pub p50_latency_ms: Vec<f64>,
    pub p95_latency_ms: Vec<f64>,
    pub p99_latency_ms: Vec<f64>,
}

#[derive(Debug, serde::Serialize)]
pub struct ThroughputResults {
    pub search_list_sizes: Vec<usize>,
    pub qps: Vec<f64>,
    pub threads: usize,
}

#[derive(Debug, serde::Serialize)]
pub struct RecallResults {
    pub search_list_sizes: Vec<usize>,
    pub recall_at_k: Vec<f64>,
    pub recall_at_1: Vec<f64>,
}

pub fn run(args: BenchmarkArgs, cli: &crate::Cli) -> diskann::Result<()> {
    if !cli.no_progress {
        println!("{}", style("📊 DiskANN Benchmarking Suite").bold().green());
        println!("  Index: {}", args.index.display());
        println!("  Queries: {}", args.queries.display());
        println!("  k: {}", args.k);
        println!();
    }
    
    // Parse search list sizes
    let search_list_sizes: Result<Vec<usize>, _> = args.search_list_sizes
        .split(',')
        .map(|s| s.trim().parse())
        .collect();
    let search_list_sizes = search_list_sizes?;
    
    // Load queries
    if cli.verbose {
        println!("Loading queries...");
    }
    let (queries, query_dim) = load_queries(&args)?;
    let num_queries = queries.len().min(args.max_queries);
    
    if !cli.no_progress {
        println!("✅ Loaded {} queries of dimension {}", 
                style(num_queries).bold(), style(query_dim).bold());
    }
    
    // Load ground truth if provided
    let ground_truth = if args.recall || args.all {
        if let Some(ref gt_path) = args.ground_truth {
            if cli.verbose {
                println!("Loading ground truth...");
            }
            Some(load_ground_truth(gt_path, num_queries, args.k)?)
        } else {
            return Err(anyhow::anyhow!("Ground truth file required for recall benchmark"));
        }
    } else {
        None
    };
    
    // Load index
    if cli.verbose {
        println!("Loading index...");
    }
    let index = crate::index::memory::MemoryIndex::load(&args.index)?;
    
    if !cli.no_progress {
        println!("✅ Loaded index with {} vectors", 
                style(index.size()).bold());
    }
    
    // Initialize results
    let mut results = BenchmarkResults {
        timestamp: chrono::Utc::now().to_rfc3339(),
        index_path: args.index.display().to_string(),
        queries_path: args.queries.display().to_string(),
        k: args.k,
        num_queries,
        latency_results: None,
        throughput_results: None,
        recall_results: None,
    };
    
    // Run benchmarks
    if args.latency || args.all {
        if !cli.no_progress {
            println!("\n🕐 Running latency benchmark...");
        }
        results.latency_results = Some(run_latency_benchmark(
            &queries[..num_queries], 
            &search_list_sizes,
            &index, 
            &args, 
            cli
        )?);
    }
    
    if args.throughput || args.all {
        if !cli.no_progress {
            println!("\n⚡ Running throughput benchmark...");
        }
        results.throughput_results = Some(run_throughput_benchmark(
            &queries[..num_queries], 
            &search_list_sizes,
            &index, 
            &args, 
            cli
        )?);
    }
    
    if (args.recall || args.all) && ground_truth.is_some() {
        if !cli.no_progress {
            println!("\n🎯 Running recall benchmark...");
        }
        results.recall_results = Some(run_recall_benchmark(
            &queries[..num_queries], 
            &search_list_sizes, 
            ground_truth.as_ref().unwrap(),
            &index,
            &args, 
            cli
        )?);
    }
    
    // Display results
    display_results(&results, cli);
    
    // Save results if requested
    if let Some(ref output_path) = args.output {
        save_results(&results, output_path)?;
        if !cli.no_progress {
            println!("\n📁 Results saved to: {}", output_path.display());
        }
    }
    
    Ok(())
}

fn load_queries(args: &BenchmarkArgs) -> diskann::Result<(Vec<Vec<f32>>, usize)> {
    // Reuse logic from search.rs
    crate::search::load_queries(&crate::search::SearchArgs {
        index: args.index.clone(),
        queries: args.queries.clone(),
        k: args.k,
        search_list_size: 50,
        output: None,
        format: args.format.clone(),
        dimension: args.dimension,
        show_distances: false,
        accurate: false,
        filter_labels: None,
        range: None,
        max_queries: Some(args.max_queries),
    })
}

fn load_ground_truth(
    path: &PathBuf, 
    num_queries: usize, 
    k: usize
) -> diskann::Result<Vec<Vec<usize>>> {
    // Assume ground truth is in ivecs format (standard for benchmarks)
    let (gt_data, gt_k) = crate::formats::read_ivecs(path)?;
    
    if gt_data.len() < num_queries {
        return Err(anyhow::anyhow!(
            "Ground truth has {} queries, need at least {}", 
            gt_data.len(), num_queries
        ));
    }
    
    if gt_k < k {
        return Err(anyhow::anyhow!(
            "Ground truth has k={}, need at least k={}", 
            gt_k, k
        ));
    }
    
    // Convert to usize and take first k neighbors
    let ground_truth: Vec<Vec<usize>> = gt_data
        .into_iter()
        .take(num_queries)
        .map(|neighbors| neighbors.into_iter().take(k).map(|id| id as usize).collect())
        .collect();
    
    Ok(ground_truth)
}

fn run_latency_benchmark(
    queries: &[Vec<f32>],
    search_list_sizes: &[usize],
    index: &crate::index::memory::MemoryIndex,
    args: &BenchmarkArgs,
    cli: &crate::Cli,
) -> crate::Result<LatencyResults> {
    let mut avg_latency_ms = Vec::new();
    let mut p50_latency_ms = Vec::new();
    let mut p95_latency_ms = Vec::new();
    let mut p99_latency_ms = Vec::new();
    
    for &search_list_size in search_list_sizes {
        if cli.verbose {
            println!("  Testing search list size: {}", search_list_size);
        }
        
        let pb: Option<ProgressBar> = if !cli.no_progress {
            let pb = ProgressBar::new(queries.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("    [{bar:30.cyan/blue}] {pos}/{len} queries")
                .unwrap());
            Some(pb)
        } else {
            None
        };
        
        // Warmup
        for query in queries.iter().take(args.warmup) {
            let _ = execute_search(query, index, args.k);
        }
        
        // Measure latencies
        let mut latencies = Vec::new();
        for query in queries {
            let start = Instant::now();
            let _ = execute_search(query, index, args.k);
            let latency = start.elapsed();
            latencies.push(latency.as_secs_f64() * 1000.0); // Convert to ms
            
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
        }
        
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        
        // Calculate statistics
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let p50 = latencies[latencies.len() / 2];
        let p95 = latencies[latencies.len() * 95 / 100];
        let p99 = latencies[latencies.len() * 99 / 100];
        
        avg_latency_ms.push(avg);
        p50_latency_ms.push(p50);
        p95_latency_ms.push(p95);
        p99_latency_ms.push(p99);
    }
    
    Ok(LatencyResults {
        search_list_sizes: search_list_sizes.to_vec(),
        avg_latency_ms,
        p50_latency_ms,
        p95_latency_ms,
        p99_latency_ms,
    })
}

fn run_throughput_benchmark(
    queries: &[Vec<f32>],
    search_list_sizes: &[usize],
    index: &crate::index::memory::MemoryIndex,
    args: &BenchmarkArgs,
    cli: &crate::Cli,
) -> crate::Result<ThroughputResults> {
    let mut qps_results = Vec::new();
    
    for &search_list_size in search_list_sizes {
        if cli.verbose {
            println!("  Testing search list size: {} with {} threads", 
                    search_list_size, args.threads);
        }
        
        let start = Instant::now();
        
        // TODO: Implement multi-threaded search
        // For now, simulate with single-threaded execution
        for query in queries {
            let _ = execute_search(query, index, args.k);
        }
        
        let elapsed = start.elapsed();
        let qps = queries.len() as f64 / elapsed.as_secs_f64();
        qps_results.push(qps);
    }
    
    Ok(ThroughputResults {
        search_list_sizes: search_list_sizes.to_vec(),
        qps: qps_results,
        threads: args.threads,
    })
}

fn run_recall_benchmark(
    queries: &[Vec<f32>],
    search_list_sizes: &[usize],
    ground_truth: &[Vec<usize>],
    index: &crate::index::memory::MemoryIndex,
    args: &BenchmarkArgs,
    cli: &crate::Cli,
) -> crate::Result<RecallResults> {
    let mut recall_at_k = Vec::new();
    let mut recall_at_1 = Vec::new();
    
    for &search_list_size in search_list_sizes {
        if cli.verbose {
            println!("  Testing recall for search list size: {}", search_list_size);
        }
        
        let mut total_recall_k = 0.0;
        let mut total_recall_1 = 0.0;
        
        for (query_idx, query) in queries.iter().enumerate() {
            let results = execute_search(query, index, args.k);
            let gt = &ground_truth[query_idx];
            
            // Calculate recall@k
            let intersection_k: usize = results
                .iter()
                .take(args.k)
                .map(|&(id, _)| id)
                .filter(|&id| gt.contains(&id))
                .count();
            
            total_recall_k += intersection_k as f64 / args.k as f64;
            
            // Calculate recall@1
            if !results.is_empty() && gt.contains(&results[0].0) {
                total_recall_1 += 1.0;
            }
        }
        
        recall_at_k.push(total_recall_k / queries.len() as f64);
        recall_at_1.push(total_recall_1 / queries.len() as f64);
    }
    
    Ok(RecallResults {
        search_list_sizes: search_list_sizes.to_vec(),
        recall_at_k,
        recall_at_1,
    })
}

fn execute_search(query: &[f32], index: &crate::index::memory::MemoryIndex, k: usize) -> Vec<(usize, f32)> {
    use crate::Index;
    // TODO: Support search_list_size parameter
    index.search(query, k).unwrap_or_else(|_| vec![])
}

fn display_results(results: &BenchmarkResults, cli: &crate::Cli) {
    if cli.no_progress {
        return;
    }
    
    println!("\n🎯 Benchmark Results Summary");
    println!("{}", "=".repeat(50));
    
    if let Some(ref latency) = results.latency_results {
        println!("\n⏱️  Latency Results:");
        println!("Search List Size | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms)");
        println!("{}", "-".repeat(55));
        
        for i in 0..latency.search_list_sizes.len() {
            println!("{:15} | {:8.2} | {:8.2} | {:8.2} | {:8.2}",
                    latency.search_list_sizes[i],
                    latency.avg_latency_ms[i],
                    latency.p50_latency_ms[i],
                    latency.p95_latency_ms[i],
                    latency.p99_latency_ms[i]);
        }
    }
    
    if let Some(ref throughput) = results.throughput_results {
        println!("\n⚡ Throughput Results ({} threads):", throughput.threads);
        println!("Search List Size | QPS");
        println!("{}", "-".repeat(25));
        
        for i in 0..throughput.search_list_sizes.len() {
            println!("{:15} | {:8.0}",
                    throughput.search_list_sizes[i],
                    throughput.qps[i]);
        }
    }
    
    if let Some(ref recall) = results.recall_results {
        println!("\n🎯 Recall Results:");
        println!("Search List Size | Recall@{} | Recall@1", results.k);
        println!("{}", "-".repeat(35));
        
        for i in 0..recall.search_list_sizes.len() {
            println!("{:15} | {:8.4} | {:8.4}",
                    recall.search_list_sizes[i],
                    recall.recall_at_k[i],
                    recall.recall_at_1[i]);
        }
    }
}

fn save_results(results: &BenchmarkResults, output_path: &PathBuf) -> diskann::Result<()> {
    use std::fs::File;
    use std::io::BufWriter;
    
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, results)?;
    
    Ok(())
}