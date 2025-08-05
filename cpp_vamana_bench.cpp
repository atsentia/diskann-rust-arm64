// Simple C++ Vamana benchmark to compare with Rust
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Simple L2 distance
float l2_distance(const vector<float>& a, const vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Find medoid using C++ DiskANN approach (O(n) centroid-based)
size_t find_medoid(const vector<vector<float>>& vectors) {
    if (vectors.empty()) return 0;
    
    size_t dimension = vectors[0].size();
    size_t num_vectors = vectors.size();
    
    // Step 1: Calculate centroid - O(n)
    vector<float> centroid(dimension, 0.0f);
    for (const auto& vec : vectors) {
        for (size_t i = 0; i < dimension; ++i) {
            centroid[i] += vec[i];
        }
    }
    for (auto& val : centroid) {
        val /= num_vectors;
    }
    
    // Step 2: Find closest point to centroid - O(n)
    float min_distance = numeric_limits<float>::max();
    size_t medoid = 0;
    
    for (size_t i = 0; i < num_vectors; ++i) {
        float distance = l2_distance(centroid, vectors[i]);
        if (distance < min_distance) {
            min_distance = distance;
            medoid = i;
        }
    }
    
    return medoid;
}

// Generate random vectors
vector<vector<float>> generate_random_vectors(size_t count, size_t dim) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    vector<vector<float>> vectors(count, vector<float>(dim));
    for (auto& vec : vectors) {
        for (auto& val : vec) {
            val = dis(gen);
        }
    }
    
    return vectors;
}

// Simple graph build simulation (sequential)
void build_graph_sequential(const vector<vector<float>>& vectors) {
    size_t n = vectors.size();
    vector<vector<size_t>> graph(n);
    
    // Find medoid
    size_t medoid = find_medoid(vectors);
    
    // Simulate graph construction (simplified)
    for (size_t i = 0; i < n; ++i) {
        // Find k nearest neighbors for each vertex (simplified)
        vector<pair<float, size_t>> distances;
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                float dist = l2_distance(vectors[i], vectors[j]);
                distances.push_back({dist, j});
            }
        }
        
        // Sort and take top k=32
        partial_sort(distances.begin(), distances.begin() + min(32UL, distances.size()), 
                    distances.end());
        
        for (size_t k = 0; k < min(32UL, distances.size()); ++k) {
            graph[i].push_back(distances[k].second);
        }
    }
}

// Parallel graph build with OpenMP
void build_graph_parallel(const vector<vector<float>>& vectors) {
    size_t n = vectors.size();
    vector<vector<size_t>> graph(n);
    
    // Find medoid
    size_t medoid = find_medoid(vectors);
    
    cout << "Building graph with " << omp_get_max_threads() << " threads" << endl;
    
    // Parallel graph construction
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t i = 0; i < n; ++i) {
        // Find k nearest neighbors for each vertex
        vector<pair<float, size_t>> distances;
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                float dist = l2_distance(vectors[i], vectors[j]);
                distances.push_back({dist, j});
            }
        }
        
        // Sort and take top k=32
        partial_sort(distances.begin(), distances.begin() + min(32UL, distances.size()), 
                    distances.end());
        
        // Update graph (in real implementation would need locks)
        for (size_t k = 0; k < min(32UL, distances.size()); ++k) {
            graph[i].push_back(distances[k].second);
        }
        
        // Progress reporting
        if (i % 1000 == 0) {
            #pragma omp critical
            {
                cout << "Progress: " << (i * 100.0 / n) << "%" << endl;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    size_t num_vectors = 25000;
    size_t dimension = 128;
    
    if (argc > 1) num_vectors = atoi(argv[1]);
    if (argc > 2) dimension = atoi(argv[2]);
    
    cout << "C++ Vamana Benchmark - ARM64" << endl;
    cout << "Testing " << num_vectors << " vectors × " << dimension << " dimensions" << endl;
    
    // Generate test data
    cout << "Generating test vectors..." << endl;
    auto vectors = generate_random_vectors(num_vectors, dimension);
    
    // Test medoid calculation
    cout << "\nTesting medoid calculation..." << endl;
    auto medoid_start = high_resolution_clock::now();
    size_t medoid = find_medoid(vectors);
    auto medoid_end = high_resolution_clock::now();
    auto medoid_time = duration_cast<microseconds>(medoid_end - medoid_start).count();
    cout << "Medoid found: " << medoid << " in " << medoid_time << " μs" << endl;
    
    // Test sequential build
    cout << "\nBuilding graph (sequential)..." << endl;
    auto seq_start = high_resolution_clock::now();
    build_graph_sequential(vectors);
    auto seq_end = high_resolution_clock::now();
    auto seq_time_ms = duration_cast<milliseconds>(seq_end - seq_start).count();
    float seq_time = seq_time_ms / 1000.0f;
    cout << "Sequential build time: " << seq_time << " seconds" << endl;
    cout << "Build rate: " << (num_vectors / seq_time) << " vectors/sec" << endl;
    
    // Test parallel build
    cout << "\nBuilding graph (parallel with OpenMP)..." << endl;
    auto par_start = high_resolution_clock::now();
    build_graph_parallel(vectors);
    auto par_end = high_resolution_clock::now();
    auto par_time_ms = duration_cast<milliseconds>(par_end - par_start).count();
    float par_time = par_time_ms / 1000.0f;
    cout << "Parallel build time: " << par_time << " seconds" << endl;
    cout << "Build rate: " << (num_vectors / par_time) << " vectors/sec" << endl;
    cout << "Speedup: " << seq_time / par_time << "x" << endl;
    
    return 0;
}