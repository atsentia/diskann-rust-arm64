//! Index abstractions for the external API

mod ann_inmem_index;
mod ann_disk_index;
mod index_impl;

pub use ann_inmem_index::{ANNInmemIndex, create_inmem_index};
pub use ann_disk_index::{ANNDiskIndex, create_disk_index, DiskIndexStorage};