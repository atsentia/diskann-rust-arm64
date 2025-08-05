//! C++ DiskANN API function signatures for drop-in compatibility

use super::*;
use std::os::raw::{c_char, c_float, c_uint, c_void};
use std::ffi::CStr;

/// Opaque handle for C++ compatibility
pub type IndexHandle = *mut c_void;

/// Create a new index
#[no_mangle]
pub extern "C" fn diskann_create_index(
    metric: c_uint,
    dimension: c_uint,
) -> IndexHandle {
    let metric = match metric {
        0 => Distance::L2,
        1 => Distance::InnerProduct,
        2 => Distance::Cosine,
        _ => return std::ptr::null_mut(),
    };
    
    let index = Box::new(DiskANNIndexWrapper {
        index: None,
        metric,
        dimension: dimension as usize,
    });
    
    Box::into_raw(index) as IndexHandle
}

/// Build index from data file
#[no_mangle]
pub unsafe extern "C" fn diskann_build_index(
    handle: IndexHandle,
    data_file: *const c_char,
    index_prefix: *const c_char,
    num_threads: c_uint,
    max_degree: c_uint,
    search_list_size: c_uint,
    alpha: c_float,
) -> c_uint {
    if handle.is_null() || data_file.is_null() || index_prefix.is_null() {
        return 1;
    }
    
    let wrapper = &mut *(handle as *mut DiskANNIndexWrapper);
    
    let data_file = match CStr::from_ptr(data_file).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };
    
    let index_prefix = match CStr::from_ptr(index_prefix).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };
    
    let params = BuildParams {
        num_threads,
        max_degree,
        search_list_size,
        alpha,
        ..Default::default()
    };
    
    match DiskANNIndex::build(data_file, index_prefix, &params, wrapper.metric) {
        Ok(index) => {
            wrapper.index = Some(index);
            0
        }
        Err(_) => 1,
    }
}

/// Load index from disk
#[no_mangle]
pub unsafe extern "C" fn diskann_load_index(
    handle: IndexHandle,
    index_prefix: *const c_char,
    num_points: c_uint,
    num_threads: c_uint,
) -> c_uint {
    if handle.is_null() || index_prefix.is_null() {
        return 1;
    }
    
    let wrapper = &mut *(handle as *mut DiskANNIndexWrapper);
    
    let index_prefix = match CStr::from_ptr(index_prefix).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };
    
    if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build_global()
            .ok();
    }
    
    match DiskANNIndex::load(
        index_prefix,
        num_points as usize,
        wrapper.dimension,
        wrapper.metric,
    ) {
        Ok(index) => {
            wrapper.index = Some(index);
            0
        }
        Err(_) => 1,
    }
}

/// Search for k nearest neighbors
#[no_mangle]
pub unsafe extern "C" fn diskann_search(
    handle: IndexHandle,
    query: *const c_float,
    k: c_uint,
    search_list_size: c_uint,
    neighbors: *mut c_uint,
    distances: *mut c_float,
) -> c_uint {
    if handle.is_null() || query.is_null() || neighbors.is_null() || distances.is_null() {
        return 0;
    }
    
    let wrapper = &*(handle as *mut DiskANNIndexWrapper);
    
    if let Some(ref index) = wrapper.index {
        let query_slice = std::slice::from_raw_parts(query, wrapper.dimension);
        let neighbors_slice = std::slice::from_raw_parts_mut(neighbors, k as usize);
        let distances_slice = std::slice::from_raw_parts_mut(distances, k as usize);
        
        let params = SearchParams {
            search_list_size,
            ..Default::default()
        };
        
        match index.search(
            query_slice,
            k as usize,
            &params,
            neighbors_slice,
            distances_slice,
        ) {
            Ok(count) => count,
            Err(_) => 0,
        }
    } else {
        0
    }
}

/// Batch search
#[no_mangle]
pub unsafe extern "C" fn diskann_batch_search(
    handle: IndexHandle,
    queries: *const c_float,
    num_queries: c_uint,
    k: c_uint,
    search_list_size: c_uint,
    neighbors: *mut c_uint,
    distances: *mut c_float,
) -> c_uint {
    if handle.is_null() || queries.is_null() || neighbors.is_null() || distances.is_null() {
        return 1;
    }
    
    let wrapper = &*(handle as *mut DiskANNIndexWrapper);
    
    if let Some(ref index) = wrapper.index {
        let queries_slice = std::slice::from_raw_parts(
            queries,
            (num_queries as usize) * wrapper.dimension,
        );
        let neighbors_slice = std::slice::from_raw_parts_mut(
            neighbors,
            (num_queries * k) as usize,
        );
        let distances_slice = std::slice::from_raw_parts_mut(
            distances,
            (num_queries * k) as usize,
        );
        
        let params = SearchParams {
            search_list_size,
            ..Default::default()
        };
        
        match index.batch_search(
            queries_slice,
            num_queries as usize,
            wrapper.dimension,
            k as usize,
            &params,
            neighbors_slice,
            distances_slice,
        ) {
            Ok(_) => 0,
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Free index
#[no_mangle]
pub unsafe extern "C" fn diskann_free_index(handle: IndexHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle as *mut DiskANNIndexWrapper);
    }
}

/// Get index stats
#[no_mangle]
pub unsafe extern "C" fn diskann_get_stats(
    handle: IndexHandle,
    stats: *mut IndexStats,
) -> c_uint {
    if handle.is_null() || stats.is_null() {
        return 1;
    }
    
    let wrapper = &*(handle as *mut DiskANNIndexWrapper);
    
    if let Some(ref index) = wrapper.index {
        *stats = index.get_stats();
        0
    } else {
        1
    }
}

// Internal wrapper structure
struct DiskANNIndexWrapper {
    index: Option<DiskANNIndex>,
    metric: Distance,
    dimension: usize,
}

/// C++ compatible build function
pub fn build_index(
    data_file: &str,
    index_prefix: &str,
    params: &BuildParams,
    metric: Distance,
) -> Result<()> {
    let index = DiskANNIndex::build(data_file, index_prefix, params, metric)?;
    
    // Index is automatically saved during build
    drop(index);
    Ok(())
}

/// C++ compatible search function
pub fn search_index(
    index_prefix: &str,
    query_file: &str,
    result_file: &str,
    num_points: usize,
    dimension: usize,
    k: usize,
    search_params: &SearchParams,
    metric: Distance,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;
    
    // Load index
    let index = DiskANNIndex::load(index_prefix, num_points, dimension, metric)?;
    
    // Load queries
    let (queries, _) = crate::formats::read_fvecs(query_file)?;
    
    // Prepare output file
    let mut output = File::create(result_file)?;
    
    // Search each query
    for query in &queries {
        let mut neighbors = vec![0u32; k];
        let mut distances = vec![0.0f32; k];
        
        let count = index.search(
            query,
            k,
            search_params,
            &mut neighbors,
            &mut distances,
        )?;
        
        // Write results in binary format
        output.write_all(&(count as u32).to_le_bytes())?;
        for i in 0..count as usize {
            output.write_all(&neighbors[i].to_le_bytes())?;
        }
        for i in 0..count as usize {
            output.write_all(&distances[i].to_le_bytes())?;
        }
    }
    
    Ok(())
}