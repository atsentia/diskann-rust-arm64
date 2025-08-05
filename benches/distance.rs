//! Benchmarks for distance calculations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use diskann::distance::{Distance, create_distance_function};
use rand::prelude::*;

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn benchmark_l2_distance(c: &mut Criterion) {
    let dimensions = vec![128, 256, 512, 768, 1024];
    
    let mut group = c.benchmark_group("l2_distance");
    
    for dim in dimensions {
        let a = generate_random_vector(dim);
        let b = generate_random_vector(dim);
        let distance_fn = create_distance_function(Distance::L2, dim);
        
        group.bench_with_input(
            BenchmarkId::new("dimension", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| {
                    distance_fn.distance(black_box(&a), black_box(&b)).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_cosine_distance(c: &mut Criterion) {
    let dimensions = vec![128, 256, 512, 768, 1024];
    
    let mut group = c.benchmark_group("cosine_distance");
    
    for dim in dimensions {
        let a = generate_random_vector(dim);
        let b = generate_random_vector(dim);
        let distance_fn = create_distance_function(Distance::Cosine, dim);
        
        group.bench_with_input(
            BenchmarkId::new("dimension", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| {
                    distance_fn.distance(black_box(&a), black_box(&b)).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_batch_distance(c: &mut Criterion) {
    let dim = 768;
    let num_points = 1000;
    
    let query = generate_random_vector(dim);
    let points: Vec<f32> = (0..num_points * dim)
        .map(|_| thread_rng().gen_range(-1.0..1.0))
        .collect();
    let mut distances = vec![0.0; num_points];
    
    let distance_fn = create_distance_function(Distance::L2, dim);
    
    c.bench_function("batch_l2_distance_1000x768", |bencher| {
        bencher.iter(|| {
            distance_fn.batch_distance(
                black_box(&query),
                black_box(&points),
                black_box(&mut distances),
            ).unwrap()
        });
    });
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn benchmark_neon_vs_scalar(c: &mut Criterion) {
    use diskann::distance::{neon::NeonDistance, scalar::ScalarDistance, DistanceFunction};
    
    let dim = 768;
    let a = generate_random_vector(dim);
    let b = generate_random_vector(dim);
    
    let neon_dist = NeonDistance::new(Distance::L2, dim);
    let scalar_dist = ScalarDistance::new(Distance::L2, dim);
    
    let mut group = c.benchmark_group("neon_vs_scalar");
    
    group.bench_function("neon_l2_768d", |bencher| {
        bencher.iter(|| {
            neon_dist.distance(black_box(&a), black_box(&b)).unwrap()
        });
    });
    
    group.bench_function("scalar_l2_768d", |bencher| {
        bencher.iter(|| {
            scalar_dist.distance(black_box(&a), black_box(&b)).unwrap()
        });
    });
    
    group.finish();
}

criterion_group!(
    distance_benches,
    benchmark_l2_distance,
    benchmark_cosine_distance,
    benchmark_batch_distance,
);

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
criterion_group!(neon_benches, benchmark_neon_vs_scalar);

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
criterion_main!(distance_benches, neon_benches);

#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
criterion_main!(distance_benches);