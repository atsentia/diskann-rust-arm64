// Dot Product Compute Shader for WebGPU
// Calculates dot product between query vector and batch of points

struct Params {
    dimension: u32,
    num_vectors: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> points: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

@compute @workgroup_size(64)
fn dot_product_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vector_idx = global_id.x;
    
    if (vector_idx >= params.num_vectors) {
        return;
    }
    
    let dimension = params.dimension;
    var dot_product: f32 = 0.0;
    
    // Calculate dot product
    for (var i: u32 = 0u; i < dimension; i++) {
        let query_val = query[i];
        let point_val = points[vector_idx * dimension + i];
        dot_product += query_val * point_val;
    }
    
    // Store negative for max-heap compatibility
    results[vector_idx] = -dot_product;
}