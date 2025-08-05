//! Benchmarks for graph operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use diskann::distance::{Distance, create_distance_function};
use diskann::graph::{VamanaGraph, SearchParams, SearchScratch, beam_search};
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

fn benchmark_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_construction");
    group.sample_size(10);
    
    for &num_vertices in &[100, 500, 1000] {
        let dimension = 128;
        let vectors = generate_random_vectors(num_vertices, dimension);
        
        group.bench_with_input(
            BenchmarkId::new("vertices", num_vertices),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let mut graph = VamanaGraph::new(
                        num_vertices,
                        dimension,
                        Distance::L2,
                        32,  // max_degree
                        50,  // search_list_size
                        1.2, // alpha
                    );
                    graph.build(black_box(vectors)).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_beam_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("beam_search");
    
    let dimension = 128;
    let num_vertices = 10_000;
    let vectors = generate_random_vectors(num_vertices, dimension);
    
    // Build graph
    let mut vamana = VamanaGraph::new(
        num_vertices,
        dimension,
        Distance::L2,
        32,
        100,
        1.2,
    );
    vamana.build(&vectors).unwrap();
    let stats = vamana.stats();
    
    // Extract graph structure for direct beam search
    let graph: Vec<Vec<usize>> = (0..num_vertices)
        .map(|_| Vec::new()) // Placeholder - would need to expose from VamanaGraph
        .collect();
    
    let distance_fn = create_distance_function(Distance::L2, dimension);
    let mut scratch = SearchScratch::new(num_vertices);
    
    // Benchmark different search list sizes
    for &search_list_size in &[50, 100, 200] {
        let params = SearchParams {
            search_list_size,
            k: 10,
            alpha: 1.2,
            use_bitvector: true,
        };
        
        let query = generate_random_vectors(1, dimension).into_iter().next().unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("search_list_size", search_list_size),
            &search_list_size,
            |b, _| {
                b.iter(|| {
                    beam_search(
                        stats.entry_point,
                        black_box(&query),
                        &params,
                        &mut scratch,
                        &graph,
                        &vectors,
                        |a, b| distance_fn.distance(a, b),
                    ).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_visited_tracking(c: &mut Criterion) {
    use diskann::graph::search::{HashSetVisited, BitVectorVisited, VisitedSet};
    
    let mut group = c.benchmark_group("visited_tracking");
    
    let num_vertices = 100_000;
    let num_visits = 1000;
    
    // Generate random visit pattern
    let mut rng = thread_rng();
    let visit_pattern: Vec<usize> = (0..num_visits)
        .map(|_| rng.gen_range(0..num_vertices))
        .collect();
    
    group.bench_function("hashset", |b| {
        let mut visited = HashSetVisited::new();
        b.iter(|| {
            visited.clear();
            for &id in &visit_pattern {
                visited.insert(black_box(id));
            }
        });
    });
    
    group.bench_function("bitvector", |b| {
        let mut visited = BitVectorVisited::new(num_vertices);
        b.iter(|| {
            visited.clear();
            for &id in &visit_pattern {
                visited.insert(black_box(id));
            }
        });
    });
    
    group.finish();
}

fn benchmark_pruning(c: &mut Criterion) {
    use diskann::graph::prune::robust_prune;
    
    let mut group = c.benchmark_group("pruning");
    
    let dimension = 128;
    let num_candidates = 100;
    let vectors = generate_random_vectors(num_candidates + 1, dimension);
    let candidates: Vec<usize> = (1..=num_candidates).collect();
    
    let distance_fn = create_distance_function(Distance::L2, dimension);
    
    for &max_degree in &[16, 32, 64] {
        group.bench_with_input(
            BenchmarkId::new("max_degree", max_degree),
            &max_degree,
            |b, &max_degree| {
                b.iter(|| {
                    robust_prune(
                        0,
                        black_box(&candidates),
                        &vectors,
                        max_degree,
                        1.2,
                        |a, b| distance_fn.distance(a, b),
                    ).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    graph_benches,
    benchmark_graph_construction,
    benchmark_beam_search,
    benchmark_visited_tracking,
    benchmark_pruning
);

criterion_main!(graph_benches);