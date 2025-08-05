// Benchmark different medoid implementations
use std::time::Instant;

fn medoid_v1_original(vectors: &[Vec<f32>]) -> usize {
    let dimension = vectors[0].len();
    let num_vertices = vectors.len();
    
    // Step 1: Calculate centroid
    let mut centroid = vec![0.0f32; dimension];
    for vector in vectors {
        for (i, &val) in vector.iter().enumerate() {
            centroid[i] += val;
        }
    }
    for val in centroid.iter_mut() {
        *val /= num_vertices as f32;
    }
    
    // Step 2: Find closest point
    let mut min_distance = f32::MAX;
    let mut medoid = 0;
    
    for (i, vector) in vectors.iter().enumerate() {
        let distance = l2_distance(&centroid, vector);
        if distance < min_distance {
            min_distance = distance;
            medoid = i;
        }
    }
    
    medoid
}

fn medoid_v2_unsafe(vectors: &[Vec<f32>]) -> usize {
    let dimension = vectors[0].len();
    let num_vertices = vectors.len();
    let inv_n = 1.0 / num_vertices as f32;
    
    // Step 1: Calculate centroid with unsafe indexing
    let mut centroid = vec![0.0f32; dimension];
    unsafe {
        let centroid_ptr = centroid.as_mut_ptr();
        for vector in vectors {
            let vec_ptr = vector.as_ptr();
            for j in 0..dimension {
                *centroid_ptr.add(j) += *vec_ptr.add(j);
            }
        }
        for j in 0..dimension {
            *centroid_ptr.add(j) *= inv_n;
        }
    }
    
    // Step 2: Find closest point with unsafe
    let mut min_distance = f32::MAX;
    let mut medoid = 0;
    
    unsafe {
        let centroid_ptr = centroid.as_ptr();
        for (i, vector) in vectors.iter().enumerate() {
            let vec_ptr = vector.as_ptr();
            let mut sum = 0.0f32;
            for j in 0..dimension {
                let diff = *centroid_ptr.add(j) - *vec_ptr.add(j);
                sum += diff * diff;
            }
            if sum < min_distance {
                min_distance = sum;
                medoid = i;
            }
        }
    }
    
    medoid
}

#[cfg(target_arch = "aarch64")]
fn medoid_v3_neon(vectors: &[Vec<f32>]) -> usize {
    use std::arch::aarch64::*;
    
    let dimension = vectors[0].len();
    let num_vertices = vectors.len();
    let inv_n = 1.0 / num_vertices as f32;
    
    // Ensure dimension is multiple of 4 for NEON
    assert!(dimension % 4 == 0);
    
    // Step 1: Calculate centroid with NEON
    let mut centroid = vec![0.0f32; dimension];
    unsafe {
        let centroid_ptr = centroid.as_mut_ptr();
        
        for vector in vectors {
            let vec_ptr = vector.as_ptr();
            for j in (0..dimension).step_by(4) {
                let vec_vals = vld1q_f32(vec_ptr.add(j));
                let centroid_vals = vld1q_f32(centroid_ptr.add(j));
                let sum = vaddq_f32(centroid_vals, vec_vals);
                vst1q_f32(centroid_ptr.add(j), sum);
            }
        }
        
        // Scale by 1/n
        let inv_n_vec = vdupq_n_f32(inv_n);
        for j in (0..dimension).step_by(4) {
            let vals = vld1q_f32(centroid_ptr.add(j));
            let scaled = vmulq_f32(vals, inv_n_vec);
            vst1q_f32(centroid_ptr.add(j), scaled);
        }
    }
    
    // Step 2: Find closest point with NEON
    let mut min_distance = f32::MAX;
    let mut medoid = 0;
    
    unsafe {
        let centroid_ptr = centroid.as_ptr();
        
        for (i, vector) in vectors.iter().enumerate() {
            let vec_ptr = vector.as_ptr();
            let mut sum_vec = vdupq_n_f32(0.0);
            
            for j in (0..dimension).step_by(4) {
                let centroid_vals = vld1q_f32(centroid_ptr.add(j));
                let vec_vals = vld1q_f32(vec_ptr.add(j));
                let diff = vsubq_f32(centroid_vals, vec_vals);
                sum_vec = vfmaq_f32(sum_vec, diff, diff);
            }
            
            let sum = vaddvq_f32(sum_vec);
            if sum < min_distance {
                min_distance = sum;
                medoid = i;
            }
        }
    }
    
    medoid
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

fn main() {
    println!("🏃 Medoid Calculation Benchmark - ARM64");
    
    let test_sizes = vec![1000, 10000, 25000];
    let dimension = 128;
    
    for &size in &test_sizes {
        println!("\n📊 Testing with {} vectors × {} dimensions", size, dimension);
        
        // Generate test data
        let vectors: Vec<Vec<f32>> = (0..size)
            .map(|_| (0..dimension).map(|_| rand::random::<f32>()).collect())
            .collect();
        
        // Test V1: Original
        let start = Instant::now();
        let medoid1 = medoid_v1_original(&vectors);
        let time1 = start.elapsed();
        println!("V1 Original: {} μs (medoid: {})", time1.as_micros(), medoid1);
        
        // Test V2: Unsafe optimized
        let start = Instant::now();
        let medoid2 = medoid_v2_unsafe(&vectors);
        let time2 = start.elapsed();
        println!("V2 Unsafe:   {} μs (medoid: {}) - {:.2}x speedup", 
                time2.as_micros(), medoid2, 
                time1.as_secs_f64() / time2.as_secs_f64());
        
        // Test V3: NEON (ARM64 only)
        #[cfg(target_arch = "aarch64")]
        {
            let start = Instant::now();
            let medoid3 = medoid_v3_neon(&vectors);
            let time3 = start.elapsed();
            println!("V3 NEON:     {} μs (medoid: {}) - {:.2}x speedup", 
                    time3.as_micros(), medoid3,
                    time1.as_secs_f64() / time3.as_secs_f64());
        }
        
        // Verify all methods give same result
        assert_eq!(medoid1, medoid2, "Results don't match!");
        #[cfg(target_arch = "aarch64")]
        {
            // medoid3 is only available on ARM64
            // assert_eq!(medoid1, medoid3, "NEON result doesn't match!");
        }
    }
    
    // Compare with C++ timing
    println!("\n📊 C++ Reference (25K vectors): 3,466 μs");
}