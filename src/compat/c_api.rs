//! C API for DiskANN - matches Microsoft DiskANN's C interface

use super::*;
use std::os::raw::{c_char, c_float, c_int, c_uint};
use std::ffi::CStr;

/// C-compatible error codes
#[repr(C)]
pub enum DiskANNErrorCode {
    Success = 0,
    InvalidParameter = -1,
    OutOfMemory = -2,
    FileNotFound = -3,
    IOError = -4,
    UnknownError = -99,
}

/// C-compatible build parameters
#[repr(C)]
pub struct DiskANNBuildParams {
    pub num_threads: c_uint,
    pub max_degree: c_uint,
    pub search_list_size: c_uint,
    pub max_occlusion_size: c_uint,
    pub alpha: c_float,
    pub saturate_graph: c_int,
    pub use_pq_build: c_int,
    pub num_pq_chunks: c_uint,
    pub use_opq: c_int,
}

/// C-compatible search parameters
#[repr(C)]
pub struct DiskANNSearchParams {
    pub search_list_size: c_uint,
    pub beamwidth: c_uint,
    pub reorder_data: c_int,
}

/// Build in-memory index
#[no_mangle]
pub unsafe extern "C" fn diskann_build_memory_index(
    data_path: *const c_char,
    metric: c_uint,
    index_path: *const c_char,
    params: *const DiskANNBuildParams,
) -> c_int {
    if data_path.is_null() || index_path.is_null() || params.is_null() {
        return DiskANNErrorCode::InvalidParameter as c_int;
    }
    
    let data_path = match CStr::from_ptr(data_path).to_str() {
        Ok(s) => s,
        Err(_) => return DiskANNErrorCode::InvalidParameter as c_int,
    };
    
    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return DiskANNErrorCode::InvalidParameter as c_int,
    };
    
    let metric = match metric {
        0 => Distance::L2,
        1 => Distance::InnerProduct,
        2 => Distance::Cosine,
        _ => return DiskANNErrorCode::InvalidParameter as c_int,
    };
    
    let params = &*params;
    let build_params = BuildParams {
        num_threads: params.num_threads,
        max_degree: params.max_degree,
        search_list_size: params.search_list_size,
        max_occlusion_size: params.max_occlusion_size,
        alpha: params.alpha,
        saturate_graph: params.saturate_graph != 0,
        use_pq_build: params.use_pq_build != 0,
        num_pq_chunks: params.num_pq_chunks,
        use_opq: params.use_opq != 0,
    };
    
    match super::cpp_api::build_index(data_path, index_path, &build_params, metric) {
        Ok(_) => DiskANNErrorCode::Success as c_int,
        Err(e) => {
            eprintln!("Build error: {}", e);
            DiskANNErrorCode::UnknownError as c_int
        }
    }
}

/// Build disk index
#[no_mangle]
pub unsafe extern "C" fn diskann_build_disk_index(
    data_path: *const c_char,
    metric: c_uint,
    index_path: *const c_char,
    params: *const DiskANNBuildParams,
    data_dimension: c_uint,
    index_ram_limit: c_uint,
    graph_ram_limit: c_uint,
) -> c_int {
    if data_path.is_null() || index_path.is_null() || params.is_null() {
        return DiskANNErrorCode::InvalidParameter as c_int;
    }
    
    let data_path = match CStr::from_ptr(data_path).to_str() {
        Ok(s) => s,
        Err(_) => return DiskANNErrorCode::InvalidParameter as c_int,
    };
    
    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return DiskANNErrorCode::InvalidParameter as c_int,
    };
    
    let metric = match metric {
        0 => Distance::L2,
        1 => Distance::InnerProduct,
        2 => Distance::Cosine,
        _ => return DiskANNErrorCode::InvalidParameter as c_int,
    };
    
    // Load vectors
    let (vectors, dimension) = match crate::formats::read_fvecs(data_path) {
        Ok((v, d)) => (v, d),
        Err(_) => {
            // Try binary format
            match crate::formats::read_binary_vectors(data_path, data_dimension as usize) {
                Ok(v) => (v, data_dimension as usize),
                Err(_) => return DiskANNErrorCode::FileNotFound as c_int,
            }
        }
    };
    
    let params = &*params;
    
    // Build PQ Flash Index
    let pq_config = PQFlashConfig {
        max_degree: params.max_degree as usize,
        search_list_size: params.search_list_size as usize,
        alpha: params.alpha,
        pq_params: crate::index::disk::PQParams {
            num_chunks: (params.num_pq_chunks as usize).max(dimension / 8),
            bits_per_chunk: 8,
        },
        num_threads: params.num_threads as usize,
        use_reorder_data: true, // Default to using reorder data
        beam_width: 4, // Default beam width
    };
    
    match PQFlashIndex::build_from_vectors(index_path, vectors, pq_config) {
        Ok(_) => DiskANNErrorCode::Success as c_int,
        Err(e) => {
            eprintln!("Build error: {}", e);
            DiskANNErrorCode::UnknownError as c_int
        }
    }
}

/// Search memory index
#[no_mangle]
pub unsafe extern "C" fn diskann_search_memory_index(
    index_path: *const c_char,
    queries: *const c_float,
    num_queries: c_uint,
    knn: c_uint,
    params: *const DiskANNSearchParams,
    gt_ids: *mut c_uint,
    gt_dists: *mut c_float,
) -> c_int {
    if index_path.is_null() || queries.is_null() || params.is_null() || 
       gt_ids.is_null() || gt_dists.is_null() {
        return DiskANNErrorCode::InvalidParameter as c_int;
    }
    
    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return DiskANNErrorCode::InvalidParameter as c_int,
    };
    
    // Load index
    let index = match crate::index::MemoryIndex::load(index_path) {
        Ok(idx) => Box::new(idx) as Box<dyn Index>,
        Err(_) => return DiskANNErrorCode::FileNotFound as c_int,
    };
    
    let params = &*params;
    let dimension = index.dimension();
    
    // Process queries
    let queries_slice = std::slice::from_raw_parts(queries, (num_queries as usize) * dimension);
    let ids_slice = std::slice::from_raw_parts_mut(gt_ids, (num_queries * knn) as usize);
    let dists_slice = std::slice::from_raw_parts_mut(gt_dists, (num_queries * knn) as usize);
    
    for i in 0..num_queries as usize {
        let query_start = i * dimension;
        let query_end = query_start + dimension;
        let query = &queries_slice[query_start..query_end];
        
        match index.search(query, knn as usize) {
            Ok(results) => {
                let result_start = i * (knn as usize);
                for (j, (id, dist)) in results.iter().enumerate() {
                    if j < knn as usize {
                        ids_slice[result_start + j] = *id as c_uint;
                        dists_slice[result_start + j] = *dist;
                    }
                }
            }
            Err(_) => return DiskANNErrorCode::UnknownError as c_int,
        }
    }
    
    DiskANNErrorCode::Success as c_int
}

/// Search disk index
#[no_mangle]
pub unsafe extern "C" fn diskann_search_disk_index(
    index_path: *const c_char,
    queries: *const c_float,
    num_queries: c_uint,
    query_dimension: c_uint,
    knn: c_uint,
    beamwidth: c_uint,
    search_ram_limit: c_uint,
    gt_ids: *mut c_uint,
    gt_dists: *mut c_float,
) -> c_int {
    if index_path.is_null() || queries.is_null() || gt_ids.is_null() || gt_dists.is_null() {
        return DiskANNErrorCode::InvalidParameter as c_int;
    }
    
    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return DiskANNErrorCode::InvalidParameter as c_int,
    };
    
    // Load PQ index - create new index and load data
    let mut index = PQFlashIndex::new(query_dimension as usize, Distance::L2, PQFlashConfig::default());
    if let Err(_) = index.load(index_path) {
        return DiskANNErrorCode::FileNotFound as c_int;
    }
    
    let dimension = query_dimension as usize;
    
    // Process queries
    let queries_slice = std::slice::from_raw_parts(queries, (num_queries as usize) * dimension);
    let ids_slice = std::slice::from_raw_parts_mut(gt_ids, (num_queries * knn) as usize);
    let dists_slice = std::slice::from_raw_parts_mut(gt_dists, (num_queries * knn) as usize);
    
    for i in 0..num_queries as usize {
        let query_start = i * dimension;
        let query_end = query_start + dimension;
        let query = &queries_slice[query_start..query_end];
        
        let search_list_size = (knn as usize) * 10; // Default search parameter
        match index.search(query, knn as usize, search_list_size) {
            Ok((results, _stats)) => {
                let result_start = i * (knn as usize);
                for (j, (id, dist)) in results.iter().enumerate() {
                    if j < knn as usize {
                        ids_slice[result_start + j] = *id as c_uint;
                        dists_slice[result_start + j] = *dist;
                    }
                }
            }
            Err(_) => return DiskANNErrorCode::UnknownError as c_int,
        }
    }
    
    DiskANNErrorCode::Success as c_int
}

/// Range search
#[no_mangle]
pub unsafe extern "C" fn diskann_range_search(
    index_path: *const c_char,
    query: *const c_float,
    query_dimension: c_uint,
    radius: c_float,
    complexity: c_uint,
    indices: *mut c_uint,
    distances: *mut c_float,
    result_count: *mut c_uint,
) -> c_int {
    if index_path.is_null() || query.is_null() || indices.is_null() || 
       distances.is_null() || result_count.is_null() {
        return DiskANNErrorCode::InvalidParameter as c_int;
    }
    
    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return DiskANNErrorCode::InvalidParameter as c_int,
    };
    
    // Load index
    let index = match crate::index::MemoryIndex::load(index_path) {
        Ok(idx) => Box::new(idx) as Box<dyn Index>,
        Err(_) => return DiskANNErrorCode::FileNotFound as c_int,
    };
    
    let query_slice = std::slice::from_raw_parts(query, query_dimension as usize);
    
    match index.range_search(query_slice, radius, complexity as usize) {
        Ok(results) => {
            *result_count = results.len() as c_uint;
            
            for (i, (id, dist)) in results.iter().enumerate() {
                if i < complexity as usize {
                    indices.add(i).write(*id as c_uint);
                    distances.add(i).write(*dist);
                }
            }
            
            DiskANNErrorCode::Success as c_int
        }
        Err(_) => DiskANNErrorCode::UnknownError as c_int,
    }
}

/// Get version string
#[no_mangle]
pub extern "C" fn diskann_get_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Set number of threads
#[no_mangle]
pub extern "C" fn diskann_set_num_threads(num_threads: c_uint) -> c_int {
    if num_threads == 0 {
        return DiskANNErrorCode::InvalidParameter as c_int;
    }
    
    match rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads as usize)
        .build_global()
    {
        Ok(_) => DiskANNErrorCode::Success as c_int,
        Err(_) => DiskANNErrorCode::UnknownError as c_int,
    }
}