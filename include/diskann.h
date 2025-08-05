/*
 * DiskANN C API Header
 * Compatible with Microsoft DiskANN C++ implementation
 */

#ifndef DISKANN_H
#define DISKANN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* Error codes */
typedef enum {
    DISKANN_SUCCESS = 0,
    DISKANN_INVALID_PARAMETER = -1,
    DISKANN_OUT_OF_MEMORY = -2,
    DISKANN_FILE_NOT_FOUND = -3,
    DISKANN_IO_ERROR = -4,
    DISKANN_UNKNOWN_ERROR = -99
} diskann_error_t;

/* Metrics */
typedef enum {
    DISKANN_METRIC_L2 = 0,
    DISKANN_METRIC_IP = 1,
    DISKANN_METRIC_COSINE = 2
} diskann_metric_t;

/* Build parameters */
typedef struct {
    uint32_t num_threads;
    uint32_t max_degree;
    uint32_t search_list_size;
    uint32_t max_occlusion_size;
    float alpha;
    int32_t saturate_graph;
    int32_t use_pq_build;
    uint32_t num_pq_chunks;
    int32_t use_opq;
} diskann_build_params_t;

/* Search parameters */
typedef struct {
    uint32_t search_list_size;
    uint32_t beamwidth;
    int32_t reorder_data;
} diskann_search_params_t;

/* Index statistics */
typedef struct {
    size_t num_points;
    size_t dimension;
    float graph_degree;
    uint32_t max_degree;
    double indexing_time;
    uint64_t memory_usage;
} diskann_index_stats_t;

/* Opaque index handle */
typedef void* diskann_index_t;

/* Version information */
const char* diskann_get_version(void);

/* Thread management */
int32_t diskann_set_num_threads(uint32_t num_threads);

/* In-memory index operations */
int32_t diskann_build_memory_index(
    const char* data_path,
    uint32_t metric,
    const char* index_path,
    const diskann_build_params_t* params
);

int32_t diskann_search_memory_index(
    const char* index_path,
    const float* queries,
    uint32_t num_queries,
    uint32_t knn,
    const diskann_search_params_t* params,
    uint32_t* gt_ids,
    float* gt_dists
);

/* Disk-based index operations */
int32_t diskann_build_disk_index(
    const char* data_path,
    uint32_t metric,
    const char* index_path,
    const diskann_build_params_t* params,
    uint32_t data_dimension,
    uint32_t index_ram_limit,
    uint32_t graph_ram_limit
);

int32_t diskann_search_disk_index(
    const char* index_path,
    const float* queries,
    uint32_t num_queries,
    uint32_t query_dimension,
    uint32_t knn,
    uint32_t beamwidth,
    uint32_t search_ram_limit,
    uint32_t* gt_ids,
    float* gt_dists
);

/* Range search */
int32_t diskann_range_search(
    const char* index_path,
    const float* query,
    uint32_t query_dimension,
    float radius,
    uint32_t complexity,
    uint32_t* indices,
    float* distances,
    uint32_t* result_count
);

/* C++ compatible API */
diskann_index_t diskann_create_index(uint32_t metric, uint32_t dimension);
void diskann_free_index(diskann_index_t handle);

uint32_t diskann_build_index(
    diskann_index_t handle,
    const char* data_file,
    const char* index_prefix,
    uint32_t num_threads,
    uint32_t max_degree,
    uint32_t search_list_size,
    float alpha
);

uint32_t diskann_load_index(
    diskann_index_t handle,
    const char* index_prefix,
    uint32_t num_points,
    uint32_t num_threads
);

uint32_t diskann_search(
    diskann_index_t handle,
    const float* query,
    uint32_t k,
    uint32_t search_list_size,
    uint32_t* neighbors,
    float* distances
);

uint32_t diskann_batch_search(
    diskann_index_t handle,
    const float* queries,
    uint32_t num_queries,
    uint32_t k,
    uint32_t search_list_size,
    uint32_t* neighbors,
    float* distances
);

uint32_t diskann_get_stats(
    diskann_index_t handle,
    diskann_index_stats_t* stats
);

#ifdef __cplusplus
}
#endif

#endif /* DISKANN_H */