# Ternary AI Inference Application - Project Summary

## Overview

A complete, self-made Rust framework for running ternary (1.58-bit) LLMs locally. No dependency on BitNet or llama.cpp for the core - they're pluggable engines.

## Project Structure

```
tern/
├── Cargo.toml                          # Workspace root
├── build.rs                            # Build script
├── config.example.toml                 # Example configuration
├── README.md                           # Original README
├── LICENSE                             # Apache-2.0
├── CONTRIBUTING.md                     # Contribution guidelines
├── GOVERNANCE.md                       # Governance model
├── CODE_OF_CONDUCT.md                  # Code of conduct
├── SECURITY.md                         # Security policy
├── CHANGELOG.md                        # Version history
├── .gitignore                          # Git ignore rules
├── .github/workflows/
│   ├── ci.yml                          # CI pipeline (Linux, Windows, macOS)
│   └── release.yml                     # Release automation
├── core/                               # Main library crate
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                      # Library entry point
│   │   ├── main.rs                     # CLI binary entry
│   │   ├── tensor/mod.rs               # Ternary tensor operations
│   │   ├── hardware/mod.rs             # Hardware detection
│   │   ├── model/mod.rs                # Model management
│   │   ├── engine/mod.rs               # Inference engine
│   │   ├── api/mod.rs                  # OpenAI-compatible API
│   │   ├── cli/mod.rs                  # Command-line interface
│   │   └── utils/mod.rs                # Utilities
├── api/                                # API server binary
│   ├── Cargo.toml
│   └── src/main.rs
├── bench/                              # Benchmark binary
│   ├── Cargo.toml
│   └── src/main.rs
├── engines/
│   ├── bitnet/                         # bitnet.cpp adapter
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   └── llamacpp/                       # llama.cpp adapter
│       ├── Cargo.toml
│       └── src/lib.rs
└── tests/
    └── integration_tests.rs            # Integration tests
```

## Key Features Implemented

### Core Tensor Operations (`core/src/tensor/mod.rs`)
- **PackedTernaryTensor**: 2-bit storage (4 weights per byte)
- **TernaryWeight**: -1, 0, +1 enum with threshold quantization
- **TernaryLinear**: Efficient addition-based matmul (no multiplication)
- **Activations**: ReLU, GELU, SiLU, Softmax
- **LayerNorm**: RMS normalization
- **Embedding**: Token embedding lookup

### Hardware Detection (`core/src/hardware/mod.rs`)
- CPU info (cores, frequency, vendor, caches)
- Memory info (total, available, used)
- OS detection
- Instruction set detection (AVX2, NEON, etc.)
- Performance tier classification (Low/Medium/High/Enthusiast)
- Model recommendations with RAM/token estimates
- Power estimation

### Model Management (`core/src/model/mod.rs`)
- ModelMetadata with full architecture config
- Safetensors loading with ternary quantization
- Hugging Face Hub integration
- Tokenizer support (tokenizers crate)
- Model caching with async RwLock
- InferenceConfig with all sampling params

### Inference Engine (`core/src/engine/mod.rs`)
- **InferenceEngine trait** for pluggable backends
- **TernaryEngine**: Pure Rust implementation
  - Full transformer forward pass
  - RoPE positional embeddings
  - Multi-head attention with KV cache
  - GQA (Grouped Query Attention) support
  - SiLU + Gate/Up/Down FFN
  - Streaming inference with callbacks
  - Sampling (temp, top-p, top-k, min-p, repetition penalty)

### API Server (`core/src/api/mod.rs`)
- OpenAI-compatible `/v1/chat/completions`
- Streaming via Server-Sent Events
- `/v1/models` listing
- `/health` endpoint
- API key authentication
- CORS support
- Model auto-loading

### CLI (`core/src/cli/mod.rs`)
- `tern chat` - Interactive/non-interactive chat
- `tern serve` - API server
- `tern download` - Model download
- `tern list` - List models/recommendations
- `tern hardware` - Show hardware info
- `tern bench` - Run benchmarks
- `tern convert` - Model conversion (placeholder)

### Engine Adapters
- **bitnet.cpp** (`engines/bitnet/`) - I2_S, TL1, TL2 kernels
- **llama.cpp** (`engines/llamacpp/`) - GGUF support

### Testing
- Unit tests in each module
- Integration tests (`tests/integration_tests.rs`)
- Hardware detection, tensor ops, engine factory, KV cache, activations, layer norm

### CI/CD
- Multi-platform (Linux, Windows, macOS)
- Lint (fmt, clippy), test, audit, docs, build
- Release automation with artifacts and checksums

## Usage

```bash
# Build
cargo build --workspace --release

# Chat
cargo run --bin tern -- chat --model bitnet-b1.58-2b-4t

# Serve API
cargo run --bin tern -- serve --port 8080

# Benchmark
cargo run --bin tern-bench -- --model bitnet-b1.58-2b-4t --runs 10

# Download model
cargo run --bin tern -- download microsoft/BitNet-b1.58-2B-4T-gguf
```

## Configuration

Copy `config.example.toml` to `config.toml` and customize:
- API bind address, port, auth
- Model cache directory
- Inference defaults (threads, context, sampling)
- Engine selection
- Hardware settings

## Dependencies

Key crates:
- `ndarray` - Tensor operations
- `tokio` - Async runtime
- `axum` - Web framework
- `hf-hub` - Hugging Face integration
- `safetensors` - Model format
- `tokenizers` - Tokenization
- `sysinfo` - Hardware detection
- `clap` - CLI parsing
- `serde` - Serialization
- `tracing` - Logging

## License

Apache-2.0