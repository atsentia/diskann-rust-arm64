//! NUMA (Non-Uniform Memory Access) awareness utilities
//!
//! This module provides Linux-specific optimizations for NUMA systems,
//! including thread pinning and memory allocation policies.

use crate::Result;
use std::collections::HashMap;
use parking_lot::RwLock;

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes available
    pub num_nodes: usize,
    /// CPU cores per NUMA node
    pub cores_per_node: Vec<Vec<usize>>,
    /// Memory available per NUMA node (in bytes)
    pub memory_per_node: Vec<u64>,
}

/// NUMA-aware thread pool configuration
#[derive(Debug, Clone)]
pub struct NumaConfig {
    /// Enable NUMA awareness
    pub enabled: bool,
    /// Preferred NUMA node for allocations
    pub preferred_node: Option<usize>,
    /// Thread pinning strategy
    pub thread_pinning: ThreadPinning,
}

/// Thread pinning strategies
#[derive(Debug, Clone)]
pub enum ThreadPinning {
    /// No thread pinning
    None,
    /// Pin threads to specific NUMA nodes
    PerNode,
    /// Pin threads to specific CPU cores
    PerCore,
}

impl Default for NumaConfig {
    fn default() -> Self {
        Self {
            enabled: is_numa_available(),
            preferred_node: None,
            thread_pinning: ThreadPinning::PerNode,
        }
    }
}

/// Global NUMA topology cache
static NUMA_TOPOLOGY: RwLock<Option<NumaTopology>> = RwLock::new(None);

/// Check if NUMA is available on this system
pub fn is_numa_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/sys/devices/system/node").exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Get NUMA topology information
pub fn get_numa_topology() -> Result<NumaTopology> {
    // Check cache first
    {
        let cached = NUMA_TOPOLOGY.read();
        if let Some(ref topology) = *cached {
            return Ok(topology.clone());
        }
    }

    // Detect topology
    let topology = detect_numa_topology()?;
    
    // Cache result
    {
        let mut cached = NUMA_TOPOLOGY.write();
        *cached = Some(topology.clone());
    }

    Ok(topology)
}

/// Detect NUMA topology on Linux systems
#[cfg(target_os = "linux")]
fn detect_numa_topology() -> Result<NumaTopology> {
    use std::fs;
    
    let node_dir = std::path::Path::new("/sys/devices/system/node");
    if !node_dir.exists() {
        return Ok(NumaTopology {
            num_nodes: 1,
            cores_per_node: vec![get_cpu_list()?],
            memory_per_node: vec![get_total_memory()?],
        });
    }

    let mut cores_per_node = Vec::new();
    let mut memory_per_node = Vec::new();
    let mut num_nodes = 0;

    // Enumerate NUMA nodes
    for entry in fs::read_dir(node_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        
        if name_str.starts_with("node") {
            if let Ok(node_id) = name_str[4..].parse::<usize>() {
                num_nodes = num_nodes.max(node_id + 1);
                
                // Read CPU list for this node
                let cpulist_path = entry.path().join("cpulist");
                let cores = if cpulist_path.exists() {
                    parse_cpu_list(&fs::read_to_string(cpulist_path)?)?
                } else {
                    Vec::new()
                };
                
                // Ensure we have enough slots
                while cores_per_node.len() <= node_id {
                    cores_per_node.push(Vec::new());
                    memory_per_node.push(0);
                }
                
                cores_per_node[node_id] = cores;
                
                // Read memory info for this node
                let meminfo_path = entry.path().join("meminfo");
                if meminfo_path.exists() {
                    memory_per_node[node_id] = parse_node_memory(&fs::read_to_string(meminfo_path)?)?;
                }
            }
        }
    }

    if num_nodes == 0 {
        // Fallback for systems without NUMA
        return Ok(NumaTopology {
            num_nodes: 1,
            cores_per_node: vec![get_cpu_list()?],
            memory_per_node: vec![get_total_memory()?],
        });
    }

    Ok(NumaTopology {
        num_nodes,
        cores_per_node,
        memory_per_node,
    })
}

/// Fallback for non-Linux systems
#[cfg(not(target_os = "linux"))]
fn detect_numa_topology() -> Result<NumaTopology> {
    Ok(NumaTopology {
        num_nodes: 1,
        cores_per_node: vec![get_cpu_list()?],
        memory_per_node: vec![get_total_memory()?],
    })
}

/// Get list of all CPU cores
fn get_cpu_list() -> Result<Vec<usize>> {
    let num_cpus = num_cpus::get();
    Ok((0..num_cpus).collect())
}

/// Get total system memory
fn get_total_memory() -> Result<u64> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    Ok(system.total_memory() * 1024) // Convert from KB to bytes
}

/// Parse CPU list format (e.g., "0-3,8-11")
#[cfg(target_os = "linux")]
fn parse_cpu_list(cpulist: &str) -> Result<Vec<usize>> {
    let mut cores = Vec::new();
    
    for range in cpulist.trim().split(',') {
        if range.contains('-') {
            let parts: Vec<&str> = range.split('-').collect();
            if parts.len() == 2 {
                let start: usize = parts[0].parse().map_err(|e| {
                    crate::Error::InvalidParameter(format!("Invalid CPU range start: {}", e))
                })?;
                let end: usize = parts[1].parse().map_err(|e| {
                    crate::Error::InvalidParameter(format!("Invalid CPU range end: {}", e))
                })?;
                cores.extend(start..=end);
            }
        } else if !range.is_empty() {
            let cpu: usize = range.parse().map_err(|e| {
                crate::Error::InvalidParameter(format!("Invalid CPU number: {}", e))
            })?;
            cores.push(cpu);
        }
    }
    
    Ok(cores)
}

/// Parse memory information from NUMA node meminfo
#[cfg(target_os = "linux")]
fn parse_node_memory(meminfo: &str) -> Result<u64> {
    for line in meminfo.lines() {
        if line.starts_with("Node") && line.contains("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(kb) = parts[2].parse::<u64>() {
                    return Ok(kb * 1024); // Convert from KB to bytes
                }
            }
        }
    }
    
    // Fallback to total system memory divided by number of nodes
    Ok(get_total_memory()?)
}

/// Pin current thread to a specific CPU core
#[cfg(target_os = "linux")]
pub fn pin_thread_to_core(core_id: usize) -> Result<()> {
    // This is a simplified implementation
    // In practice, you'd use libc::sched_setaffinity or a crate like `hwloc`
    // For now, we'll provide a placeholder that could be extended
    
    log::debug!("Would pin thread to core {}", core_id);
    Ok(())
}

/// Pin current thread to a specific NUMA node
#[cfg(target_os = "linux")]
pub fn pin_thread_to_node(node_id: usize) -> Result<()> {
    let topology = get_numa_topology()?;
    
    if node_id >= topology.num_nodes {
        return Err(crate::Error::InvalidParameter(
            format!("Invalid NUMA node {}, only {} nodes available", node_id, topology.num_nodes)
        ).into());
    }
    
    if !topology.cores_per_node[node_id].is_empty() {
        // Pin to first core in the node as a simple strategy
        let core_id = topology.cores_per_node[node_id][0];
        pin_thread_to_core(core_id)?;
    }
    
    Ok(())
}

/// Placeholder for non-Linux systems
#[cfg(not(target_os = "linux"))]
pub fn pin_thread_to_core(_core_id: usize) -> Result<()> {
    log::debug!("Thread pinning not supported on this platform");
    Ok(())
}

/// Placeholder for non-Linux systems
#[cfg(not(target_os = "linux"))]
pub fn pin_thread_to_node(_node_id: usize) -> Result<()> {
    log::debug!("NUMA thread pinning not supported on this platform");
    Ok(())
}

/// Allocate memory with NUMA locality preferences
pub fn numa_alloc(_size: usize, _preferred_node: Option<usize>) -> Result<Vec<u8>> {
    // This is a placeholder for NUMA-aware allocation
    // In practice, you'd use libc::mbind or similar system calls
    // For now, fall back to regular allocation
    
    log::debug!("NUMA-aware allocation not yet implemented, using standard allocation");
    Ok(Vec::new())
}

/// NUMA-aware worker pool for parallel operations
pub struct NumaWorkerPool {
    config: NumaConfig,
    topology: NumaTopology,
    workers_per_node: HashMap<usize, Vec<std::thread::JoinHandle<()>>>,
}

impl NumaWorkerPool {
    /// Create a new NUMA-aware worker pool
    pub fn new(config: NumaConfig) -> Result<Self> {
        let topology = get_numa_topology()?;
        
        Ok(Self {
            config,
            topology,
            workers_per_node: HashMap::new(),
        })
    }
    
    /// Get optimal number of workers for the system
    pub fn optimal_worker_count(&self) -> usize {
        if self.config.enabled {
            // Use all available cores across all NUMA nodes
            self.topology.cores_per_node.iter()
                .map(|cores| cores.len())
                .sum()
        } else {
            num_cpus::get()
        }
    }
    
    /// Get NUMA node for optimal memory allocation
    pub fn preferred_node_for_allocation(&self, _size: usize) -> Option<usize> {
        if self.config.enabled {
            self.config.preferred_node.or_else(|| {
                // Choose node with most available memory
                self.topology.memory_per_node
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &memory)| memory)
                    .map(|(node, _)| node)
            })
        } else {
            None
        }
    }
}

/// Initialize NUMA optimizations for the current process
pub fn init_numa_optimizations(config: &NumaConfig) -> Result<()> {
    if !config.enabled {
        log::info!("NUMA optimizations disabled");
        return Ok(());
    }
    
    if !is_numa_available() {
        log::warn!("NUMA not available on this system");
        return Ok(());
    }
    
    let topology = get_numa_topology()?;
    log::info!("NUMA topology detected: {} nodes", topology.num_nodes);
    
    for (node_id, cores) in topology.cores_per_node.iter().enumerate() {
        log::debug!("Node {}: {} cores, {} MB memory", 
                   node_id, 
                   cores.len(), 
                   topology.memory_per_node[node_id] / (1024 * 1024));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_detection() {
        let topology = get_numa_topology().unwrap();
        assert!(topology.num_nodes >= 1);
        assert!(!topology.cores_per_node.is_empty());
        assert!(!topology.memory_per_node.is_empty());
    }

    #[test]
    fn test_numa_config_default() {
        let config = NumaConfig::default();
        // Should work on any system
        assert!(matches!(config.thread_pinning, ThreadPinning::PerNode));
    }

    #[test]
    fn test_worker_pool_creation() {
        let config = NumaConfig::default();
        let pool = NumaWorkerPool::new(config).unwrap();
        let worker_count = pool.optimal_worker_count();
        assert!(worker_count > 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_cpu_list_parsing() {
        let cores = parse_cpu_list("0-3,8-11").unwrap();
        assert_eq!(cores, vec![0, 1, 2, 3, 8, 9, 10, 11]);
        
        let single = parse_cpu_list("5").unwrap();
        assert_eq!(single, vec![5]);
        
        let mixed = parse_cpu_list("0,2-4,7").unwrap();
        assert_eq!(mixed, vec![0, 2, 3, 4, 7]);
    }

    #[test]
    fn test_thread_pinning_no_crash() {
        // These should not crash, even if they don't do anything on non-Linux
        let _ = pin_thread_to_core(0);
        let _ = pin_thread_to_node(0);
    }
}