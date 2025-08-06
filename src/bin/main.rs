//! DiskANN Rust Command-Line Interface
//!
//! This CLI provides comprehensive tools for building, searching, and analyzing
//! DiskANN indices with various configurations and optimizations.

use clap::{Parser, Subcommand};
use console::style;
use std::path::PathBuf;

use diskann::cli::{build, search, benchmark, convert, info};

#[derive(Parser)]
#[command(name = "diskann")]
#[command(about = "DiskANN Rust - High-performance vector search")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(author = "DiskANN Rust Contributors")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
    
    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,
    
    /// Disable progress bars and use simple text output
    #[arg(long, global = true)]
    pub no_progress: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Build an index from vectors
    Build(build::BuildArgs),
    
    /// Search an existing index
    Search(search::SearchArgs),
    
    /// Run benchmarks on indices
    Benchmark(benchmark::BenchmarkArgs),
    
    /// Convert between vector formats
    Convert(convert::ConvertArgs),
    
    /// Show information about indices and vectors
    Info(info::InfoArgs),
}

fn main() -> diskann::Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging based on verbosity
    if cli.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }
    
    // Print banner
    if !cli.no_progress {
        println!("{}", style("🚀 DiskANN Rust CLI").bold().blue());
        println!("{}", style("High-performance vector search with ARM64 NEON optimizations").dim());
        println!();
    }
    
    // Convert CLI to library format
    let lib_cli = diskann::Cli {
        verbose: cli.verbose,
        no_progress: cli.no_progress,
    };
    
    // Execute command
    match cli.command {
        Commands::Build(args) => build::run(args, &lib_cli),
        Commands::Search(args) => search::run(args, &lib_cli),
        Commands::Benchmark(args) => benchmark::run(args, &lib_cli),
        Commands::Convert(args) => convert::run(args, &lib_cli),
        Commands::Info(args) => info::run(args, &lib_cli),
    }
}