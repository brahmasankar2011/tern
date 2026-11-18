// Build script for Ternary LLM Inference

fn main() {
    // Tell cargo to re-run if these files change
    println!("cargo:rerun-if-changed=config.example.toml");
    println!("cargo:rerun-if-changed=README.md");

    // Set version from Cargo.toml
    let version = env!("CARGO_PKG_VERSION");
    println!("cargo:rustc-env=TERN_VERSION={}", version);

    // Enable optimizations for release
    if std::env::var("PROFILE").unwrap_or_default() == "release" {
        println!("cargo:rustc-cfg=optimize_for_size");
    }
}