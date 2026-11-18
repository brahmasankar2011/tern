//! Ternary Benchmark Binary

use tern_core::{ModelManager, HardwareDetector, EngineFactory, engine::InferenceConfig};
use anyhow::Result;
use clap::Parser;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "tern-bench", version, about = "Ternary LLM Benchmark")]
struct Args {
    /// Model to benchmark
    #[arg(short, long)]
    model: String,

    /// Engine to benchmark
    #[arg(short, long, default_value = "ternary")]
    engine: String,

    /// Number of tokens to generate
    #[arg(long, default_value = "100")]
    tokens: usize,

    /// Number of warmup runs
    #[arg(long, default_value = "3")]
    warmup: usize,

    /// Number of benchmark runs
    #[arg(long, default_value = "10")]
    runs: usize,

    /// Prompt to use
    #[arg(long, default_value = "The quick brown fox jumps over the lazy dog.")]
    prompt: String,

    /// Model cache directory
    #[arg(long, default_value = "./models")]
    cache_dir: PathBuf,

    /// Output results as JSON
    #[arg(long)]
    json: bool,

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

    // Load model
    let config = InferenceConfig {
        threads: 0, // Auto
        ..Default::default()
    };
    let model = model_manager.load_model(&args.model, Some(config)).await?;

    // Create engine
    let mut engine = EngineFactory::create(&args.engine, config)?;
    engine.initialize(model).await?;

    // Run benchmark
    let results = run_benchmark(&mut engine, &args).await?;

    // Print results
    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        print_results(&results);
    }

    engine.shutdown().await?;
    Ok(())
}

/// Benchmark results
#[derive(Debug, serde::Serialize)]
struct BenchmarkResults {
    model: String,
    engine: String,
    hardware: String,
    runs: usize,
    tokens_per_run: usize,
    mean_ms: f32,
    median_ms: f32,
    min_ms: f32,
    max_ms: f32,
    std_dev_ms: f32,
    mean_tok_s: f32,
    estimated_power_w: f32,
    energy_per_token_j: f32,
}

/// Run benchmark
async fn run_benchmark(
    engine: &mut Box<dyn tern_core::engine::InferenceEngine>,
    args: &Args,
) -> Result<BenchmarkResults> {
    use std::time::Instant;

    let prompt = &args.prompt;
    let tokens = tern_core::api::tokenize_simple(prompt);

    let input = tern_core::engine::InferenceInput {
        tokens: tokens.clone(),
        config: tern_core::model::InferenceConfig::default(),
        kv_cache: None,
    };

    // Warmup
    info!("Warming up ({} runs)...", args.warmup);
    for _ in 0..args.warmup {
        let _ = engine.infer(&input).await?;
    }

    // Benchmark
    info!("Benchmarking ({} runs)...", args.runs);
    let mut times = Vec::new();
    let mut total_tokens = 0;

    for i in 0..args.runs {
        let start = Instant::now();
        let output = engine.infer(&input).await?;
        let elapsed = start.elapsed();

        times.push(elapsed.as_secs_f32() * 1000.0); // ms
        total_tokens += output.tokens_processed;

        if args.runs <= 10 {
            println!("Run {}: {:.2} ms ({:.1} tok/s)",
                i + 1, elapsed.as_secs_f32() * 1000.0,
                output.tokens_processed as f32 / elapsed.as_secs_f32());
        }
    }

    // Statistics
    let mean_time = times.iter().sum::<f32>() / times.len() as f32;
    let mut sorted_times = times.clone();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_time = sorted_times[sorted_times.len() / 2];
    let min_time = *sorted_times.first().unwrap();
    let max_time = *sorted_times.last().unwrap();

    // Standard deviation
    let variance: f32 = times.iter().map(|&t| (t - mean_time).powi(2)).sum::<f32>() / times.len() as f32;
    let std_dev = variance.sqrt();

    let mean_tok_s = total_tokens as f32 / (mean_time * args.runs as f32 / 1000.0);

    // Energy estimate (simplified)
    let power = 50.0; // Would use actual hardware detection
    let energy_per_token = power / mean_tok_s;

    Ok(BenchmarkResults {
        model: args.model.clone(),
        engine: args.engine.clone(),
        hardware: "auto-detected".to_string(),
        runs: args.runs,
        tokens_per_run: args.tokens,
        mean_ms: mean_time,
        median_ms: median_time,
        min_ms: min_time,
        max_ms: max_time,
        std_dev_ms: std_dev,
        mean_tok_s,
        estimated_power_w: power,
        energy_per_token_j: energy_per_token,
    })
}

/// Print benchmark results
fn print_results(results: &BenchmarkResults) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Benchmark Results                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Model              │ {:<40} ║", results.model);
    println!("║ Engine             │ {:<40} ║", results.engine);
    println!("║ Runs               │ {:<40} ║", results.runs);
    println!("║ Tokens per run     │ {:<40} ║", results.tokens_per_run);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Mean Time          │ {:<40} ║", format!("{:.2} ms", results.mean_ms));
    println!("║ Median Time        │ {:<40} ║", format!("{:.2} ms", results.median_ms));
    println!("║ Min Time           │ {:<40} ║", format!("{:.2} ms", results.min_ms));
    println!("║ Max Time           │ {:<40} ║", format!("{:.2} ms", results.max_ms));
    println!("║ Std Dev            │ {:<40} ║", format!("{:.2} ms", results.std_dev_ms));
    println!("║ Mean Throughput    │ {:<40} ║", format!("{:.1} tok/s", results.mean_tok_s));
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Est. Power         │ {:<40} ║", format!("{:.1} W", results.estimated_power_w));
    println!("║ Energy per Token   │ {:<40} ║", format!("{:.3} J/tok", results.energy_per_token_j));
    println!("╚══════════════════════════════════════════════════════════════╝");
}