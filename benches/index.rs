//! Benchmarks for index operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use diskann::{Distance, IndexBuilder};
use rand::prelude::*;

fn generate_random_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

fn benchmark_index_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_build");
    group.sample_size(10); // Reduce sample size for expensive operations
    
    for &num_vectors in &[100, 500, 1000] {
        for &dimension in &[64, 128, 256] {
            let vectors = generate_random_vectors(num_vectors, dimension);
            
            group.bench_with_input(
                BenchmarkId::new(format!("{}d", dimension), num_vectors),
                &vectors,
                |b, vectors| {
                    b.iter(|| {
                        IndexBuilder::new()
                            .dimensions(dimension)
                            .metric(Distance::L2)
                            .max_degree(32)
                            .search_list_size(50)
                            .build_from_vectors(black_box(vectors.clone()))
                            .unwrap()
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_index_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_search");
    
    // Build indices once
    let dimensions = vec![128, 256, 768];
    let num_vectors = 10_000;
    
    for dim in dimensions {
        let vectors = generate_random_vectors(num_vectors, dim);
        let index = IndexBuilder::new()
            .dimensions(dim)
            .metric(Distance::L2)
            .max_degree(32)
            .search_list_size(100)
            .build_from_vectors(vectors.clone())
            .unwrap();
        
        // Prepare query vectors
        let queries: Vec<Vec<f32>> = (0..100)
            .map(|_| generate_random_vectors(1, dim).into_iter().next().unwrap())
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("dimension", dim),
            &dim,
            |b, _| {
                let mut query_idx = 0;
                b.iter(|| {
                    let results = index.search(black_box(&queries[query_idx]), 10).unwrap();
                    query_idx = (query_idx + 1) % queries.len();
                    results
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_search_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_scalability");
    group.sample_size(20);
    
    let dimension = 128;
    let k_values = vec![1, 5, 10, 50, 100];
    
    // Build a large index
    let vectors = generate_random_vectors(50_000, dimension);
    let index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::L2)
        .max_degree(32)
        .search_list_size(200)
        .build_from_vectors(vectors.clone())
        .unwrap();
    
    let query = generate_random_vectors(1, dimension).into_iter().next().unwrap();
    
    for &k in &k_values {
        group.bench_with_input(
            BenchmarkId::new("k", k),
            &k,
            |b, &k| {
                b.iter(|| {
                    index.search(black_box(&query), black_box(k)).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_different_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("metric_comparison");
    
    let dimension = 256;
    let num_vectors = 5_000;
    let vectors = generate_random_vectors(num_vectors, dimension);
    
    // Build indices for each metric
    let l2_index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::L2)
        .build_from_vectors(vectors.clone())
        .unwrap();
    
    let cosine_index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::Cosine)
        .build_from_vectors(vectors.clone())
        .unwrap();
    
    let ip_index = IndexBuilder::new()
        .dimensions(dimension)
        .metric(Distance::InnerProduct)
        .build_from_vectors(vectors.clone())
        .unwrap();
    
    let query = generate_random_vectors(1, dimension).into_iter().next().unwrap();
    
    group.bench_function("L2", |b| {
        b.iter(|| l2_index.search(black_box(&query), 10).unwrap())
    });
    
    group.bench_function("Cosine", |b| {
        b.iter(|| cosine_index.search(black_box(&query), 10).unwrap())
    });
    
    group.bench_function("InnerProduct", |b| {
        b.iter(|| ip_index.search(black_box(&query), 10).unwrap())
    });
    
    group.finish();
}

criterion_group!(
    index_benches,
    benchmark_index_build,
    benchmark_index_search,
    benchmark_search_scalability,
    benchmark_different_metrics
);

criterion_main!(index_benches);