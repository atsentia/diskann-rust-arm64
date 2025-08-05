//! Comprehensive Performance Benchmarking Suite
//!
//! This benchmark suite measures DiskANN performance across various
//! configurations and workloads.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use diskann::{Distance, IndexBuilder, DynamicIndex, PQFlashIndex, PQFlashConfig};
use rand::Rng;
use std::time::Duration;

/// Generate random vectors for benchmarking
fn generate_random_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

/// Normalize vectors for cosine similarity
fn normalize_vectors(vectors: &mut [Vec<f32>]) {
    for vector in vectors {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector {
                *x /= norm;
            }
        }
    }
}

/// Benchmark index construction
fn bench_index_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_build");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    
    for size in [1000, 5000, 10000].iter() {
        for dim in [128, 256, 768].iter() {
            let vectors = generate_random_vectors(*size, *dim);
            
            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(
                BenchmarkId::new("in_memory", format!("{}x{}", size, dim)),
                &vectors,
                |b, vectors| {
                    b.iter(|| {
                        let index = IndexBuilder::new()
                            .dimensions(*dim)
                            .metric(Distance::L2)
                            .max_degree(64)
                            .search_list_size(100)
                            .build_from_vectors(black_box(vectors.clone()))
                            .unwrap();
                        black_box(index);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark search performance
fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    
    // Test different index sizes and dimensions
    let configs = vec![
        (1000, 128, 10),   // Small index
        (10000, 256, 10),  // Medium index
        (10000, 768, 10),  // Large dimension
    ];
    
    for (size, dim, k) in configs {
        let vectors = generate_random_vectors(size, dim);
        let index = IndexBuilder::new()
            .dimensions(dim)
            .metric(Distance::L2)
            .max_degree(64)
            .search_list_size(100)
            .build_from_vectors(vectors.clone())
            .unwrap();
        
        let queries = generate_random_vectors(100, dim);
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_query", format!("{}x{}_k{}", size, dim, k)),
            &(index, queries[0].clone()),
            |b, (index, query)| {
                b.iter(|| {
                    let results = index.search(black_box(query), k).unwrap();
                    black_box(results);
                });
            },
        );
        
        // Batch search
        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_100", format!("{}x{}_k{}", size, dim, k)),
            &(index, queries),
            |b, (index, queries)| {
                b.iter(|| {
                    for query in queries {
                        let results = index.search(black_box(query), k).unwrap();
                        black_box(results);
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark distance calculations
fn bench_distance_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");
    
    for dim in [128, 256, 512, 768, 1024, 2048].iter() {
        let query = generate_random_vectors(1, *dim)[0].clone();
        let points = generate_random_vectors(1000, *dim);
        let flat_points: Vec<f32> = points.into_iter().flatten().collect();
        let mut distances = vec![0.0; 1000];
        
        // L2 distance
        let l2_fn = diskann::create_distance_function(Distance::L2, *dim);
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("l2_batch_1000", format!("dim{}", dim)),
            &(query.clone(), flat_points.clone(), distances.clone()),
            |b, (query, points, mut dists)| {
                b.iter(|| {
                    l2_fn.batch_distance(
                        black_box(query),
                        black_box(points),
                        black_box(&mut dists),
                    ).unwrap();
                });
            },
        );
        
        // Cosine distance
        let mut norm_query = vec![query.clone()];
        normalize_vectors(&mut norm_query);
        let query_norm = norm_query[0].clone();
        
        let cosine_fn = diskann::create_distance_function(Distance::Cosine, *dim);
        group.bench_with_input(
            BenchmarkId::new("cosine_batch_1000", format!("dim{}", dim)),
            &(query_norm, flat_points.clone(), distances.clone()),
            |b, (query, points, mut dists)| {
                b.iter(|| {
                    cosine_fn.batch_distance(
                        black_box(query),
                        black_box(points),
                        black_box(&mut dists),
                    ).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark dynamic index operations
fn bench_dynamic_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_index");
    group.sample_size(50);
    
    let dim = 256;
    let initial_size = 5000;
    
    // Build initial index
    let initial_vectors = generate_random_vectors(initial_size, dim);
    let mut index = DynamicIndex::build_from_vectors(
        initial_vectors,
        dim,
        Distance::L2,
        64,
        100,
        1.2,
    ).unwrap();
    
    // Benchmark insertions
    let new_vectors = generate_random_vectors(100, dim);
    group.throughput(Throughput::Elements(1));
    group.bench_function("insert_single", |b| {
        let mut i = 0;
        b.iter(|| {
            let vector = &new_vectors[i % new_vectors.len()];
            let id = index.insert(black_box(vector.clone())).unwrap();
            black_box(id);
            i += 1;
        });
    });
    
    // Benchmark deletions
    group.bench_function("delete_single", |b| {
        let mut i = 0;
        b.iter(|| {
            let id = i % initial_size;
            index.delete(black_box(id)).unwrap();
            i += 1;
        });
    });
    
    // Benchmark search on dynamic index
    let query = generate_random_vectors(1, dim)[0].clone();
    group.bench_function("search_dynamic", |b| {
        b.iter(|| {
            let results = index.search(black_box(&query), 10).unwrap();
            black_box(results);
        });
    });
    
    group.finish();
}

/// Benchmark memory vs disk performance
fn bench_memory_vs_disk(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_vs_disk");
    group.sample_size(10);
    
    let size = 10000;
    let dim = 768;
    let k = 10;
    
    // Generate test data
    let vectors = generate_random_vectors(size, dim);
    let queries = generate_random_vectors(10, dim);
    
    // In-memory index
    let mem_index = IndexBuilder::new()
        .dimensions(dim)
        .metric(Distance::Cosine)
        .max_degree(64)
        .search_list_size(100)
        .build_from_vectors(vectors.clone())
        .unwrap();
    
    group.throughput(Throughput::Elements(queries.len() as u64));
    group.bench_function("memory_search", |b| {
        b.iter(|| {
            for query in &queries {
                let results = mem_index.search(black_box(query), k).unwrap();
                black_box(results);
            }
        });
    });
    
    // Disk-based index (if temp file creation works)
    if let Ok(temp_dir) = tempfile::tempdir() {
        let index_path = temp_dir.path().join("test.pq.idx");
        
        let config = PQFlashConfig {
            dimension: dim,
            metric: Distance::Cosine,
            num_chunks: 96,
            bits_per_chunk: 8,
            search_cache_size: 1000,
            reorder_data: true,
        };
        
        if let Ok(mut disk_index) = PQFlashIndex::build_from_vectors(
            index_path.to_str().unwrap(),
            vectors,
            config,
        ) {
            group.bench_function("disk_search", |b| {
                b.iter(|| {
                    for query in &queries {
                        let results = disk_index.search(black_box(query), k).unwrap();
                        black_box(results);
                    }
                });
            });
        }
    }
    
    group.finish();
}

/// Benchmark filtered search
fn bench_filtered_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtered_search");
    
    let size = 10000;
    let dim = 256;
    let k = 10;
    
    // Generate vectors with labels
    let vectors = generate_random_vectors(size, dim);
    let mut rng = rand::thread_rng();
    let labels: Vec<Vec<u32>> = (0..size)
        .map(|_| {
            // Each vector has 1-5 random labels from 0-99
            let num_labels = rng.gen_range(1..=5);
            (0..num_labels)
                .map(|_| rng.gen_range(0..100))
                .collect()
        })
        .collect();
    
    // Build index with labels
    let index = IndexBuilder::new()
        .dimensions(dim)
        .metric(Distance::L2)
        .max_degree(64)
        .search_list_size(100)
        .with_labels(labels)
        .build_from_vectors(vectors)
        .unwrap();
    
    let query = generate_random_vectors(1, dim)[0].clone();
    
    // No filter (baseline)
    group.bench_function("no_filter", |b| {
        b.iter(|| {
            let results = index.search(black_box(&query), k).unwrap();
            black_box(results);
        });
    });
    
    // Single label filter
    let filter = diskann::LabelFilter::Any(vec![42]);
    group.bench_function("single_label", |b| {
        b.iter(|| {
            let results = index.search_with_filter(
                black_box(&query),
                k,
                black_box(filter.clone()),
            ).unwrap();
            black_box(results);
        });
    });
    
    // Multiple label filter
    let filter = diskann::LabelFilter::Any(vec![10, 20, 30, 40, 50]);
    group.bench_function("multi_label", |b| {
        b.iter(|| {
            let results = index.search_with_filter(
                black_box(&query),
                k,
                black_box(filter.clone()),
            ).unwrap();
            black_box(results);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_index_build,
    bench_search,
    bench_distance_functions,
    bench_dynamic_index,
    bench_memory_vs_disk,
    bench_filtered_search
);

criterion_main!(benches);