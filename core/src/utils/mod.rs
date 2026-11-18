//! Utility functions and helpers

use anyhow::Result;
use std::path::Path;

/// Format bytes as human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit_idx = 0;

    while value >= 1024.0 && unit_idx < UNITS.len() - 1 {
        value /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.2} {}", value, UNITS[unit_idx])
    }
}

/// Format duration as human-readable string
pub fn format_duration(duration: std::time::Duration) -> String {
    let millis = duration.as_millis();
    if millis < 1000 {
        format!("{} ms", millis)
    } else if millis < 60_000 {
        format!("{:.2} s", millis as f64 / 1000.0)
    } else if millis < 3_600_000 {
        format!("{:.2} min", millis as f64 / 60_000.0)
    } else {
        format!("{:.2} h", millis as f64 / 3_600_000.0)
    }
}

/// Format tokens per second
pub fn format_tokens_per_sec(tokens_per_sec: f32) -> String {
    if tokens_per_sec >= 1000.0 {
        format!("{:.1}k tok/s", tokens_per_sec / 1000.0)
    } else {
        format!("{:.1} tok/s", tokens_per_sec)
    }
}

/// Validate model ID format
pub fn validate_model_id(model_id: &str) -> Result<()> {
    if model_id.is_empty() {
        anyhow::bail!("Model ID cannot be empty");
    }
    if model_id.len() > 200 {
        anyhow::bail!("Model ID too long");
    }
    // Basic validation - alphanumeric, dash, underscore, dot, slash
    for c in model_id.chars() {
        if !c.is_ascii_alphanumeric() && !matches!(c, '-' | '_' | '.' | '/') {
            anyhow::bail!("Model ID contains invalid character: {}", c);
        }
    }
    Ok(())
}

/// Get default cache directory
pub fn default_cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap())
        .join("tern")
        .join("models")
}

/// Ensure directory exists
pub fn ensure_dir(path: &Path) -> Result<()> {
    if !path.exists() {
        std::fs::create_dir_all(path)
            .with_context(|| format!("Failed to create directory: {}", path.display()))?;
    }
    Ok(())
}

/// Calculate file checksum (SHA256)
pub fn file_checksum(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Progress bar helper
pub struct ProgressBar {
    total: u64,
    current: u64,
    prefix: String,
}

impl ProgressBar {
    pub fn new(total: u64, prefix: impl Into<String>) -> Self {
        Self {
            total,
            current: 0,
            prefix: prefix.into(),
        }
    }

    pub fn increment(&mut self, delta: u64) {
        self.current = (self.current + delta).min(self.total);
        self.draw();
    }

    pub fn set(&mut self, current: u64) {
        self.current = current.min(self.total);
        self.draw();
    }

    fn draw(&self) {
        let percent = if self.total > 0 {
            self.current as f64 / self.total as f64 * 100.0
        } else {
            0.0
        };

        let bar_width = 40;
        let filled = (bar_width as f64 * percent / 100.0) as usize;
        let empty = bar_width - filled;

        let bar = "█".repeat(filled) + &"░".repeat(empty);
        print!("\r{} [{}] {:.1}% ({}/{})", self.prefix, bar, percent, self.current, self.total);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }

    pub fn finish(&self) {
        println!();
    }
}

/// Async file downloader with progress
pub async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    progress: Option<&mut ProgressBar>,
) -> Result<()> {
    let response = client.get(url).send().await?;
    let total_size = response.content_length().unwrap_or(0);

    if let Some(pb) = progress {
        pb.total = total_size;
        pb.current = 0;
    }

    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = response.bytes_stream();

    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        if let Some(pb) = progress {
            pb.increment(chunk.len() as u64);
        }
    }

    file.flush().await?;
    if let Some(pb) = progress {
        pb.finish();
    }

    Ok(())
}

/// Parse comma-separated values
pub fn parse_csv_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Truncate string to max length with ellipsis
pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        ".".repeat(max_len)
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(std::time::Duration::from_millis(500)), "500 ms");
        assert_eq!(format_duration(std::time::Duration::from_secs(2)), "2.00 s");
        assert_eq!(format_duration(std::time::Duration::from_secs(90)), "1.50 min");
    }

    #[test]
    fn test_validate_model_id() {
        assert!(validate_model_id("user/model").is_ok());
        assert!(validate_model_id("model-name_v2.1").is_ok());
        assert!(validate_model_id("").is_err());
        assert!(validate_model_id("model@name").is_err());
    }

    #[test]
    fn test_parse_csv_list() {
        assert_eq!(parse_csv_list("a,b,c"), vec!["a", "b", "c"]);
        assert_eq!(parse_csv_list(" a , b , c "), vec!["a", "b", "c"]);
        assert_eq!(parse_csv_list(""), vec![]);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello...");
        assert_eq!(truncate("hi", 2), "..");
    }
}