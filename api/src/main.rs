//! Ternary API Server Binary

use tern_api::{run_server, ApiConfig};
use tern_core::{ModelManager, HardwareDetector};
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "tern-api", version, about = "Ternary LLM API Server")]
struct Args {
    /// Bind address
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// API key
    #[arg(long)]
    api_key: Option<String>,

    /// Engine to use
    #[arg(short, long, default_value = "ternary")]
    engine: String,

    /// Default model
    #[arg(long)]
    model: Option<String>,

    /// Enable CORS
    #[arg(long, default_value = "true")]
    cors: bool,

    /// Model cache directory
    #[arg(long, default_value = "./models")]
    cache_dir: PathBuf,

    /// Verbosity
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Setup logging
    let level = match args.verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .init();

    // Detect hardware
    let mut detector = HardwareDetector::new();
    let hardware = detector.detect();
    info!("Detected hardware: {} cores, {} GB RAM", hardware.cpu.physical_cores, hardware.memory.total_bytes / (1024*1024*1024));

    // Create model manager
    let model_manager = Arc::new(ModelManager::new(args.cache_dir, hardware)?);

    // Pre-load default model if specified
    if let Some(model_id) = args.model {
        info!("Pre-loading model: {}", model_id);
        let _ = model_manager.load_model(&model_id, None).await;
    }

    let config = ApiConfig {
        bind_address: args.host,
        port: args.port,
        api_key: args.api_key,
        cors: args.cors,
        timeout_seconds: 300,
        max_concurrent_requests: 100,
    };

    run_server(config, model_manager, &args.engine).await
}