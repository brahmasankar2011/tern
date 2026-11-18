//! Command-line interface

use crate::api::{run_server, ApiConfig};
use crate::engine::{EngineFactory, InferenceConfig, InferenceEngine};
use crate::hardware::HardwareDetector;
use crate::model::{ModelManager, InferenceConfig as ModelInferenceConfig};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// Ternary LLM Inference Application
#[derive(Parser, Debug)]
#[command(name = "tern", version, about = "Ternary (1.58-bit) LLM Inference", long_about = None)]
pub struct Cli {
    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Configuration file
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Model cache directory
    #[arg(long, value_name = "DIR", default_value = "./models")]
    pub cache_dir: PathBuf,

    #[command(subcommand)]
    pub command: Commands,
}

/// Subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run inference chat session
    Chat(ChatArgs),
    /// Run API server
    Serve(ServeArgs),
    /// Download a model
    Download(DownloadArgs),
    /// List available/downloaded models
    List(ListArgs),
    /// Show hardware info and recommendations
    Hardware(HardwareArgs),
    /// Run benchmark
    Bench(BenchArgs),
    /// Convert model to ternary format
    Convert(ConvertArgs),
}

/// Chat arguments
#[derive(Parser, Debug)]
pub struct ChatArgs {
    /// Model ID to use
    #[arg(short, long, value_name = "MODEL")]
    pub model: Option<String>,

    /// Engine to use
    #[arg(short, long, value_name = "ENGINE", default_value = "ternary")]
    pub engine: EngineName,

    /// Temperature
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Top-p
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Top-k
    #[arg(long, default_value = "50")]
    pub top_k: usize,

    /// Max tokens to generate
    #[arg(long, default_value = "1024")]
    pub max_tokens: usize,

    /// Number of threads
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// Context length
    #[arg(long, default_value = "4096")]
    pub context_length: usize,

    /// Random seed
    #[arg(long)]
    pub seed: Option<u64>,

    /// System prompt
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Interactive mode
    #[arg(short, long, default_value = "true")]
    pub interactive: bool,

    /// Initial prompt (for non-interactive)
    #[arg(value_name = "PROMPT")]
    pub prompt: Option<String>,
}

/// Serve arguments
#[derive(Parser, Debug)]
pub struct ServeArgs {
    /// Bind address
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// API key
    #[arg(long)]
    pub api_key: Option<String>,

    /// Engine to use
    #[arg(short, long, value_name = "ENGINE", default_value = "ternary")]
    pub engine: EngineName,

    /// Default model
    #[arg(long)]
    pub model: Option<String>,

    /// Enable CORS
    #[arg(long, default_value = "true")]
    pub cors: bool,

    /// Request timeout (seconds)
    #[arg(long, default_value = "300")]
    pub timeout: u64,

    /// Max concurrent requests
    #[arg(long, default_value = "100")]
    pub max_concurrent: usize,
}

/// Download arguments
#[derive(Parser, Debug)]
pub struct DownloadArgs {
    /// Model ID to download
    #[arg(value_name = "MODEL_ID")]
    pub model_id: String,

    /// Revision/branch
    #[arg(long)]
    pub revision: Option<String>,

    /// Force re-download
    #[arg(long)]
    pub force: bool,
}

/// List arguments
#[derive(Parser, Debug)]
pub struct ListArgs {
    /// Show only downloaded models
    #[arg(long)]
    pub local: bool,

    /// Show recommended models for current hardware
    #[arg(long)]
    pub recommended: bool,
}

/// Hardware arguments
#[derive(Parser, Debug)]
pub struct HardwareArgs {
    /// Output format
    #[arg(short, long, value_enum, default_value = "table")]
    pub format: OutputFormat,
}

/// Benchmark arguments
#[derive(Parser, Debug)]
pub struct BenchArgs {
    /// Model to benchmark
    #[arg(short, long)]
    pub model: String,

    /// Engine to benchmark
    #[arg(short, long, value_name = "ENGINE", default_value = "ternary")]
    pub engine: EngineName,

    /// Number of tokens to generate
    #[arg(long, default_value = "100")]
    pub tokens: usize,

    /// Number of warmup runs
    #[arg(long, default_value = "3")]
    pub warmup: usize,

    /// Number of benchmark runs
    #[arg(long, default_value = "10")]
    pub runs: usize,

    /// Prompt to use
    #[arg(long, default_value = "The quick brown fox jumps over the lazy dog.")]
    pub prompt: String,

    /// Output format
    #[arg(short, long, value_enum, default_value = "table")]
    pub format: OutputFormat,
}

/// Convert arguments
#[derive(Parser, Debug)]
pub struct ConvertArgs {
    /// Input model path
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output directory
    #[arg(value_name = "OUTPUT")]
    pub output: PathBuf,

    /// Quantization threshold
    #[arg(long, default_value = "0.02")]
    pub threshold: f32,

    /// Target architecture
    #[arg(long, value_enum, default_value = "bitnet")]
    pub target: ConvertTarget,
}

/// Engine name enum
#[derive(Debug, Clone, ValueEnum)]
pub enum EngineName {
    Ternary,
    Bitnet,
    Llamacpp,
}

impl From<EngineName> for &'static str {
    fn from(val: EngineName) -> Self {
        match val {
            EngineName::Ternary => "ternary",
            EngineName::Bitnet => "bitnet",
            EngineName::Llamacpp => "llamacpp",
        }
    }
}

/// Convert target
#[derive(Debug, Clone, ValueEnum)]
pub enum ConvertTarget {
    Bitnet,
    Generic,
}

/// Output format
#[derive(Debug, Clone, ValueEnum)]
pub enum OutputFormat {
    Table,
    Json,
    Csv,
}

/// Run the CLI
pub async fn run(cli: Cli) -> Result<()> {
    // Setup logging
    let level = match cli.verbose {
        0 => Level::WARN,
        1 => Level::INFO,
        2 => Level::DEBUG,
        _ => Level::TRACE,
    };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Detect hardware
    let mut detector = HardwareDetector::new();
    let hardware = detector.detect();
    info!("Detected hardware: {} cores, {} GB RAM", hardware.cpu.physical_cores, hardware.memory.total_bytes / (1024*1024*1024));

    // Create model manager
    let model_manager = Arc::new(ModelManager::new(cli.cache_dir.clone(), hardware.clone())?);

    match cli.command {
        Commands::Chat(args) => run_chat(model_manager, hardware, args).await,
        Commands::Serve(args) => run_serve(model_manager, hardware, args).await,
        Commands::Download(args) => run_download(model_manager, args).await,
        Commands::List(args) => run_list(model_manager, hardware, args).await,
        Commands::Hardware(args) => run_hardware(hardware, args).await,
        Commands::Bench(args) => run_bench(model_manager, hardware, args).await,
        Commands::Convert(args) => run_convert(args).await,
    }
}

/// Run chat session
async fn run_chat(
    model_manager: Arc<ModelManager>,
    hardware: crate::hardware::HardwareInfo,
    args: ChatArgs,
) -> Result<()> {
    // Get model recommendations if no model specified
    let model_id = if let Some(m) = args.model {
        m
    } else {
        let recommendations = model_manager.recommend_models();
        if let Some(rec) = recommendations.first() {
            info!("Auto-selected model: {} ({}B params)", rec.model_id, rec.size_b);
            rec.model_id.clone()
        } else {
            anyhow::bail!("No models available and no model specified");
        }
    };

    // Load model
    let config = ModelInferenceConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        max_tokens: args.max_tokens,
        threads: args.threads.unwrap_or(hardware.recommended_threads),
        context_length: args.context_length,
        seed: args.seed,
        ..Default::default()
    };

    info!("Loading model: {}", model_id);
    let model = model_manager.load_model(&model_id, Some(config)).await?;

    // Create engine
    let mut engine = EngineFactory::create(args.engine.into(), config.clone())?;
    engine.initialize(model).await?;

    // Chat loop
    if args.interactive {
        run_interactive_chat(engine, args.system_prompt).await?;
    } else if let Some(prompt) = args.prompt {
        run_single_prompt(engine, prompt).await?;
    }

    engine.shutdown().await?;
    Ok(())
}

/// Run interactive chat
async fn run_interactive_chat(
    mut engine: Box<dyn InferenceEngine>,
    system_prompt: Option<String>,
) -> Result<()> {
    use std::io::{self, Write};

    println!("Ternary Chat - Type 'exit' or 'quit' to exit");
    println!("------------------------------------------------");

    let mut conversation = Vec::new();
    if let Some(sys) = system_prompt {
        conversation.push(crate::api::ChatMessage {
            role: "system".to_string(),
            content: sys,
            name: None,
        });
    }

    loop {
        print!("\n> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if matches!(input, "exit" | "quit" | "/exit" | "/quit") {
            break;
        }

        if input == "/clear" {
            conversation.clear();
            if let Some(sys) = &system_prompt {
                conversation.push(crate::api::ChatMessage {
                    role: "system".to_string(),
                    content: sys.clone(),
                    name: None,
                });
            }
            println!("Conversation cleared.");
            continue;
        }

        conversation.push(crate::api::ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
            name: None,
        });

        // Convert to tokens (simplified)
        let prompt = crate::api::messages_to_prompt(&conversation);
        let tokens = crate::api::tokenize_simple(&prompt);

        let input = crate::engine::InferenceInput {
            tokens,
            config: crate::model::InferenceConfig::default(),
            kv_cache: None,
        };

        // Stream response
        print!("Assistant: ");
        io::stdout().flush()?;

        let mut full_response = String::new();
        engine.infer_stream(&input, Box::new(move |chunk| {
            if let Some(token) = chunk.token {
                let text = crate::api::decode_token(token);
                print!("{}", text);
                io::stdout().flush().unwrap();
                full_response.push_str(&text);
            }
            Ok(())
        })).await?;

        println!();
        conversation.push(crate::api::ChatMessage {
            role: "assistant".to_string(),
            content: full_response,
            name: None,
        });
    }

    println!("\nGoodbye!");
    Ok(())
}

/// Run single prompt
async fn run_single_prompt(
    mut engine: Box<dyn InferenceEngine>,
    prompt: String,
) -> Result<()> {
    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: prompt,
        name: None,
    }];

    let prompt_text = crate::api::messages_to_prompt(&messages);
    let tokens = crate::api::tokenize_simple(&prompt_text);

    let input = crate::engine::InferenceInput {
        tokens,
        config: crate::model::InferenceConfig::default(),
        kv_cache: None,
    };

    let output = engine.infer(&input).await?;
    let text = crate::api::decode_simple(&output.logits);
    println!("{}", text);

    Ok(())
}

/// Run API server
async fn run_serve(
    model_manager: Arc<ModelManager>,
    hardware: crate::hardware::HardwareInfo,
    args: ServeArgs,
) -> Result<()> {
    let config = ApiConfig {
        bind_address: args.host,
        port: args.port,
        api_key: args.api_key,
        cors: args.cors,
        timeout_seconds: args.timeout,
        max_concurrent_requests: args.max_concurrent,
    };

    // Pre-load default model if specified
    if let Some(model_id) = args.model {
        info!("Pre-loading model: {}", model_id);
        let _ = model_manager.load_model(&model_id, None).await;
    }

    run_server(config, model_manager, args.engine.into()).await
}

/// Run download
async fn run_download(
    model_manager: Arc<ModelManager>,
    args: DownloadArgs,
) -> Result<()> {
    info!("Downloading model: {}", args.model_id);
    let _ = model_manager.load_model(&args.model_id, None).await?;
    info!("Model downloaded successfully");
    Ok(())
}

/// Run list
async fn run_list(
    model_manager: Arc<ModelManager>,
    hardware: crate::hardware::HardwareInfo,
    args: ListArgs,
) -> Result<()> {
    if args.recommended {
        let recommendations = model_manager.recommend_models();
        println!("Recommended models for your hardware:");
        println!("{:<30} {:>6} {:>10} {:>12} {}", "Model", "Size", "RAM (GB)", "Est. tok/s", "Fits");
        for rec in recommendations {
            println!("{:<30} {:>6}B {:>10.1} {:>12.1} {}",
                rec.model_id, rec.size_b, rec.estimated_ram_gb, rec.estimated_tokens_per_sec,
                if rec.fits_in_memory { "✓" } else { "✗" });
        }
        return Ok(());
    }

    let loaded = model_manager.list_loaded_models().await;
    if loaded.is_empty() {
        println!("No models loaded. Use 'tern download <model>' to download a model.");
    } else {
        println!("Loaded models:");
        for model in loaded {
            println!("  - {}", model);
        }
    }

    Ok(())
}

/// Run hardware info
async fn run_hardware(
    hardware: crate::hardware::HardwareInfo,
    args: HardwareArgs,
) -> Result<()> {
    match args.format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&hardware)?);
        }
        OutputFormat::Csv => {
            println!("property,value");
            println!("cpu_brand,{}", hardware.cpu.brand);
            println!("physical_cores,{}", hardware.cpu.physical_cores);
            println!("logical_cores,{}", hardware.cpu.logical_cores);
            println!("total_ram_gb,{:.1}", hardware.memory.total_bytes as f64 / (1024.0*1024.0*1024.0));
            println!("available_ram_gb,{:.1}", hardware.memory.available_bytes as f64 / (1024.0*1024.0*1024.0));
            println!("os,{}", hardware.os.name);
            println!("arch,{}", hardware.os.arch);
            println!("performance_tier,{:?}", hardware.performance_tier);
            println!("recommended_threads,{}", hardware.recommended_threads);
        }
        OutputFormat::Table => {
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║                    Hardware Information                        ║");
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║ CPU Brand           │ {:<40} ║", hardware.cpu.brand);
            println!("║ Physical Cores      │ {:<40} ║", hardware.cpu.physical_cores);
            println!("║ Logical Cores       │ {:<40} ║", hardware.cpu.logical_cores);
            println!("║ Base Frequency      │ {:<40} ║", format!("{} MHz", hardware.cpu.base_frequency_mhz));
            println!("║ Vendor              │ {:<40} ║", hardware.cpu.vendor_id);
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║ Total RAM           │ {:<40} ║", format!("{:.1} GB", hardware.memory.total_bytes as f64 / (1024.0*1024.0*1024.0)));
            println!("║ Available RAM       │ {:<40} ║", format!("{:.1} GB", hardware.memory.available_bytes as f64 / (1024.0*1024.0*1024.0)));
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║ OS                  │ {:<40} ║", hardware.os.name);
            println!("║ Version             │ {:<40} ║", hardware.os.version);
            println!("║ Architecture        │ {:<40} ║", hardware.os.arch);
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║ Performance Tier    │ {:<40} ║", format!("{:?}", hardware.performance_tier));
            println!("║ Recommended Threads │ {:<40} ║", hardware.recommended_threads);
            println!("║ Instruction Sets    │ {:<40} ║", hardware.instruction_sets.join(", "));
            println!("╚══════════════════════════════════════════════════════════════╝");
        }
    }
    Ok(())
}

/// Run benchmark
async fn run_bench(
    model_manager: Arc<ModelManager>,
    hardware: crate::hardware::HardwareInfo,
    args: BenchArgs,
) -> Result<()> {
    info!("Running benchmark for model: {}", args.model);

    let config = ModelInferenceConfig {
        threads: hardware.recommended_threads,
        ..Default::default()
    };

    let model = model_manager.load_model(&args.model, Some(config)).await?;
    let mut engine = EngineFactory::create(args.engine.into(), config.clone())?;
    engine.initialize(model).await?;

    let prompt = args.prompt;
    let tokens = crate::api::tokenize_simple(&prompt);

    let input = crate::engine::InferenceInput {
        tokens: tokens.clone(),
        config: config.clone(),
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
        let start = std::time::Instant::now();
        let output = engine.infer(&input).await?;
        let elapsed = start.elapsed();

        times.push(elapsed.as_secs_f32());
        total_tokens += output.tokens_processed;

        if args.runs <= 10 {
            println!("Run {}: {:.2} ms ({:.1} tok/s)",
                i + 1, elapsed.as_secs_f32() * 1000.0,
                output.tokens_processed as f32 / elapsed.as_secs_f32());
        }
    }

    // Statistics
    let avg_time = times.iter().sum::<f32>() / times.len() as f32;
    let min_time = times.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_time = times.iter().cloned().fold(0.0f32, f32::max);
    let avg_toks = total_tokens as f32 / (avg_time * args.runs as f32);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Benchmark Results                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Model              │ {:<40} ║", args.model);
    println!("║ Engine             │ {:<40} ║", args.engine.into());
    println!("║ Runs               │ {:<40} ║", args.runs);
    println!("║ Tokens per run     │ {:<40} ║", args.tokens);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Avg Time           │ {:<40} ║", format!("{:.2} ms", avg_time * 1000.0));
    println!("║ Min Time           │ {:<40} ║", format!("{:.2} ms", min_time * 1000.0));
    println!("║ Max Time           │ {:<40} ║", format!("{:.2} ms", max_time * 1000.0));
    println!("║ Avg Throughput     │ {:<40} ║", format!("{:.1} tok/s", avg_toks));
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Energy estimate
    let power = crate::hardware::HardwareDetector::new().estimate_power_watts(&hardware, config.threads, 7);
    let energy_per_token = power / avg_toks;
    println!("\nEstimated power: {:.1} W", power);
    println!("Estimated energy per token: {:.3} J/tok", energy_per_token);

    engine.shutdown().await?;
    Ok(())
}

/// Run convert
async fn run_convert(args: ConvertArgs) -> Result<()> {
    info!("Converting model: {} -> {}", args.input.display(), args.output.display());
    // Placeholder for model conversion logic
    anyhow::bail!("Model conversion not yet implemented");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::try_parse_from(["tern", "chat", "--model", "test-model"]).unwrap();
        assert!(matches!(cli.command, Commands::Chat(_)));
    }

    #[test]
    fn test_engine_name_conversion() {
        assert_eq!(<&str>::from(EngineName::Ternary), "ternary");
        assert_eq!(<&str>::from(EngineName::Bitnet), "bitnet");
        assert_eq!(<&str>::from(EngineName::Llamacpp), "llamacpp");
    }
}