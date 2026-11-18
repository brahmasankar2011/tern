//! Ternary LLM Inference Core Library
//!
//! This crate provides the core functionality for running ternary (1.58-bit)
//! large language models on CPU with high efficiency.

pub mod api;
pub mod cli;
pub mod engine;
pub mod hardware;
pub mod model;
pub mod tensor;
pub mod utils;

use anyhow::Result;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the library
pub fn init() -> Result<()> {
    // Initialize logging, etc.
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}