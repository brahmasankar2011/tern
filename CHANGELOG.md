# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with workspace
- Core tensor operations for ternary (1.58-bit) inference
- Hardware detection and model recommendations
- Model management with Hugging Face integration
- Pluggable inference engine interface
- Built-in ternary inference engine
- OpenAI-compatible API server
- Command-line interface (chat, serve, download, bench)
- BitNet.cpp engine adapter (placeholder)
- llama.cpp engine adapter (placeholder)
- Benchmark harness
- Comprehensive test suite
- CI/CD workflows for Linux, Windows, macOS
- Governance, contributing, security, and code of conduct docs

## [0.1.0] - 2026-11-18

### Added
- Initial pre-alpha release
- Ternary tensor operations (packed 2-bit storage)
- TernaryLinear layer with efficient addition-based matmul
- LayerNorm, Embedding, and activation functions (ReLU, GELU, SiLU, Softmax)
- Hardware detection (CPU, memory, OS, instruction sets)
- Performance tier classification (Low/Medium/High/Enthusiast)
- Model recommendations based on hardware
- Model loading from safetensors with ternary quantization
- Tokenizer support via tokenizers crate
- Inference engine trait with streaming support
- KV cache for attention
- Sampling (temperature, top-p, top-k, min-p, repetition penalty)
- RoPE positional embeddings
- Causal attention mask
- OpenAI-compatible /v1/chat/completions endpoint
- Streaming responses via SSE
- Model listing and health check endpoints
- CLI commands: chat, serve, download, list, hardware, bench, convert
- Configuration file support
- Progress bars for downloads
- Unit and integration tests