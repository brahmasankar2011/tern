//! Ternary LLM Inference - Main Binary

use tern_core::cli::{Cli, run};
use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    run(cli).await
}