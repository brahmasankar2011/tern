# Contributing to Tern

Thank you for your interest in contributing to Tern! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Rust (stable toolchain) - install via [rustup](https://rustup.rs)
- Git
- CMake 3.22+ (for engine building)
- Clang 18+ (for bitnet.cpp)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/tern/tern.git
cd tern

# Build the core library
cargo build --workspace

# Run tests
cargo test --workspace

# Run the CLI
cargo run --bin tern -- --help
```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/tern/tern/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, CPU, RAM)
   - Logs if applicable

### Suggesting Features

1. Check existing [Discussions](https://github.com/tern/tern/discussions) and issues
2. Create a new discussion or issue with:
   - Clear description of the feature
   - Use cases and motivation
   - Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `cargo test --workspace`
5. Run clippy: `cargo clippy --workspace -- -D warnings`
6. Format code: `cargo fmt --all`
7. Sign your commits: `git commit -s -m "your message"`
8. Push to your fork and open a PR

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat(engine): add support for TL2 kernel

Adds the TL2 kernel implementation for improved
inference performance on AVX2-capable CPUs.

Closes #123
```

## Development Guidelines

### Code Style

- Follow Rust standard style (use `cargo fmt`)
- Pass `cargo clippy` with no warnings
- Write meaningful comments for public APIs
- Use `todo!()` for incomplete implementations

### Testing

- Write unit tests for new functionality
- Add integration tests for user-facing features
- Test on multiple platforms when possible
- Benchmark performance-critical code

### Architecture

- Keep the engine interface clean and abstract
- Prefer composition over inheritance
- Use async/await for I/O operations
- Minimize dependencies in core library

### Performance

- Profile before optimizing
- Use criterion for benchmarks
- Consider SIMD for hot paths
- Minimize allocations in inference loop

## Project Structure

```
tern/
├── core/                 # Core library
│   ├── src/
│   │   ├── tensor/       # Tensor operations
│   │   ├── hardware/     # Hardware detection
│   │   ├── model/        # Model management
│   │   ├── engine/       # Inference engine
│   │   ├── api/          # API server
│   │   ├── cli/          # Command-line interface
│   │   └── utils/        # Utilities
├── engines/
│   ├── bitnet/           # bitnet.cpp adapter
│   └── llamacpp/         # llama.cpp adapter
├── api/                  # API server binary
├── bench/                # Benchmark harness
└── tests/                # Integration tests
```

## Review Process

1. All PRs require at least one review
2. CI must pass (tests, clippy, fmt)
3. Maintainers will review within 7 days
4. Address review comments promptly
5. Squash commits before merge if needed

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.