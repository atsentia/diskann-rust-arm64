//! Common types and utilities for the external API

use std::fmt;
use std::error::Error;

/// Result type for ANN operations
pub type ANNResult<T> = Result<T, ANNError>;

/// Error type for ANN operations
#[derive(Debug)]
pub enum ANNError {
    /// General index error
    IndexError(String),
    /// I/O error
    IOError(std::io::Error),
    /// Invalid parameter error
    InvalidParameter(String),
    /// Not implemented error
    NotImplemented(String),
}

impl ANNError {
    /// Log and create an index error
    pub fn log_index_error(msg: String) -> Self {
        log::error!("{}", msg);
        ANNError::IndexError(msg)
    }
}

impl fmt::Display for ANNError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ANNError::IndexError(msg) => write!(f, "Index error: {}", msg),
            ANNError::IOError(err) => write!(f, "I/O error: {}", err),
            ANNError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            ANNError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
        }
    }
}

impl Error for ANNError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ANNError::IOError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ANNError {
    fn from(err: std::io::Error) -> Self {
        ANNError::IOError(err)
    }
}

impl From<anyhow::Error> for ANNError {
    fn from(err: anyhow::Error) -> Self {
        ANNError::IndexError(err.to_string())
    }
}

impl From<crate::Error> for ANNError {
    fn from(err: crate::Error) -> Self {
        ANNError::IndexError(err.to_string())
    }
}