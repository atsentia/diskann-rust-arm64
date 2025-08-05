// L2 Distance Compute Shader for WebGPU
// Calculates squared L2 distance between query vector and batch of points

struct Params {
    dimension: u32,
    num_vectors: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> points: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

@compute @workgroup_size(64)
fn l2_distance_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vector_idx = global_id.x;
    
    if (vector_idx >= params.num_vectors) {
        return;
    }
    
    let dimension = params.dimension;
    var sum: f32 = 0.0;
    
    // Calculate L2 squared distance
    for (var i: u32 = 0u; i < dimension; i++) {
        let query_val = query[i];
        let point_val = points[vector_idx * dimension + i];
        let diff = query_val - point_val;
        sum += diff * diff;
    }
    
    // Store sqrt for L2 distance (GPU sqrt is fast)
    results[vector_idx] = sqrt(sum);
}