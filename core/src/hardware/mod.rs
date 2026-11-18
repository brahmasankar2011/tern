//! Hardware detection and capability assessment

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use sysinfo::{CpuExt, System, SystemExt};

/// Detected hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU information
    pub cpu: CpuInfo,
    /// Memory information
    pub memory: MemoryInfo,
    /// OS information
    pub os: OsInfo,
    /// Available instruction sets
    pub instruction_sets: Vec<String>,
    /// Recommended thread count for inference
    pub recommended_threads: usize,
    /// Estimated performance tier
    pub performance_tier: PerformanceTier,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// CPU brand/model name
    pub brand: String,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores (threads)
    pub logical_cores: usize,
    /// Base frequency in MHz
    pub base_frequency_mhz: u64,
    /// Max frequency in MHz
    pub max_frequency_mhz: u64,
    /// Vendor ID
    pub vendor_id: String,
    /// L1 cache size (KB)
    pub l1_cache_kb: usize,
    /// L2 cache size (KB)
    pub l2_cache_kb: usize,
    /// L3 cache size (KB)
    pub l3_cache_kb: usize,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total system memory in bytes
    pub total_bytes: u64,
    /// Available memory in bytes
    pub available_bytes: u64,
    /// Used memory in bytes
    pub used_bytes: u64,
    /// Memory speed in MT/s (if available)
    pub speed_mt_s: Option<u32>,
}

/// OS information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsInfo {
    /// OS name
    pub name: String,
    /// OS version
    pub version: String,
    /// Kernel version
    pub kernel_version: String,
    /// Architecture
    pub arch: String,
}

/// Performance tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceTier {
    /// Low-end: 2-4 cores, <8GB RAM
    Low,
    /// Mid-range: 4-8 cores, 8-16GB RAM
    Medium,
    /// High-end: 8-16 cores, 16-32GB RAM
    High,
    /// Enthusiast: 16+ cores, 32GB+ RAM
    Enthusiast,
}

impl PerformanceTier {
    /// Get recommended max model size in parameters (billions)
    pub fn max_model_size_b(&self) -> usize {
        match self {
            PerformanceTier::Low => 2,
            PerformanceTier::Medium => 7,
            PerformanceTier::High => 14,
            PerformanceTier::Enthusiast => 32,
        }
    }

    /// Get recommended quantization
    pub fn recommended_quant(&self) -> &'static str {
        match self {
            PerformanceTier::Low => "1.58-bit (ternary)",
            PerformanceTier::Medium => "1.58-bit (ternary)",
            PerformanceTier::High => "1.58-bit (ternary) or 4-bit",
            PerformanceTier::Enthusiast => "1.58-bit (ternary) or 4-bit",
        }
    }
}

/// Model recommendation based on hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    /// Model identifier
    pub model_id: String,
    /// Model size in billions of parameters
    pub size_b: usize,
    /// Estimated RAM usage in GB
    pub estimated_ram_gb: f32,
    /// Estimated tokens/sec on this hardware
    pub estimated_tokens_per_sec: f32,
    /// Whether this model fits in available memory
    pub fits_in_memory: bool,
    /// Confidence score (0-1)
    pub confidence: f32,
}

/// Hardware detector
pub struct HardwareDetector {
    system: System,
}

impl HardwareDetector {
    /// Create a new hardware detector
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        Self { system }
    }

    /// Detect all hardware information
    pub fn detect(&mut self) -> HardwareInfo {
        self.system.refresh_all();

        let cpu_info = self.detect_cpu();
        let memory_info = self.detect_memory();
        let os_info = self.detect_os();
        let instruction_sets = self.detect_instruction_sets(&cpu_info);
        let recommended_threads = self.calculate_recommended_threads(&cpu_info);
        let performance_tier = self.classify_performance_tier(&cpu_info, &memory_info);

        HardwareInfo {
            cpu: cpu_info,
            memory: memory_info,
            os: os_info,
            instruction_sets,
            recommended_threads,
            performance_tier,
        }
    }

    /// Detect CPU information
    fn detect_cpu(&self) -> CpuInfo {
        let cpus = self.system.cpus();
        let cpu = cpus.first().cloned().unwrap_or_default();

        // Count physical vs logical cores
        let physical_cores = self.system.physical_core_count().unwrap_or(cpus.len());
        let logical_cores = cpus.len();

        CpuInfo {
            brand: cpu.brand().to_string(),
            physical_cores,
            logical_cores,
            base_frequency_mhz: cpu.frequency(),
            max_frequency_mhz: cpu.frequency(), // sysinfo doesn't easily provide max freq
            vendor_id: cpu.vendor_id().to_string(),
            l1_cache_kb: 0, // Would need platform-specific code
            l2_cache_kb: 0,
            l3_cache_kb: 0,
        }
    }

    /// Detect memory information
    fn detect_memory(&self) -> MemoryInfo {
        let total = self.system.total_memory();
        let available = self.system.available_memory();
        let used = self.system.used_memory();

        MemoryInfo {
            total_bytes: total * 1024, // sysinfo returns KB
            available_bytes: available * 1024,
            used_bytes: used * 1024,
            speed_mt_s: None,
        }
    }

    /// Detect OS information
    fn detect_os(&self) -> OsInfo {
        OsInfo {
            name: System::name().unwrap_or_else(|| "Unknown".to_string()),
            version: System::os_version().unwrap_or_else(|| "Unknown".to_string()),
            kernel_version: System::kernel_version().unwrap_or_else(|| "Unknown".to_string()),
            arch: std::env::consts::ARCH.to_string(),
        }
    }

    /// Detect available instruction sets
    fn detect_instruction_sets(&self, cpu: &CpuInfo) -> Vec<String> {
        let mut sets = Vec::new();

        // Check for common instruction sets based on vendor and architecture
        #[cfg(target_arch = "x86_64")]
        {
            sets.push("x86_64".to_string());
            sets.push("SSE".to_string());
            sets.push("SSE2".to_string());
            sets.push("SSE3".to_string());
            sets.push("SSSE3".to_string());
            sets.push("SSE4.1".to_string());
            sets.push("SSE4.2".to_string());
            sets.push("AVX".to_string());
            sets.push("AVX2".to_string());
            // AVX-512 detection would need cpuid
            if cpu.brand.to_lowercase().contains("intel") {
                sets.push("AVX-512F".to_string()); // Might be present
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            sets.push("ARM64".to_string());
            sets.push("NEON".to_string());
            sets.push("SVE".to_string()); // Might be present
        }

        sets
    }

    /// Calculate recommended thread count for inference
    fn calculate_recommended_threads(&self, cpu: &CpuInfo) -> usize {
        // Use physical cores for compute-bound inference
        // Leave 1-2 cores for system
        let threads = cpu.physical_cores.saturating_sub(1);
        threads.max(1).min(cpu.logical_cores)
    }

    /// Classify performance tier
    fn classify_performance_tier(&self, cpu: &CpuInfo, memory: &MemoryInfo) -> PerformanceTier {
        let total_ram_gb = memory.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let cores = cpu.physical_cores;

        match (cores, total_ram_gb) {
            (c, r) if c >= 16 && r >= 32.0 => PerformanceTier::Enthusiast,
            (c, r) if c >= 8 && r >= 16.0 => PerformanceTier::High,
            (c, r) if c >= 4 && r >= 8.0 => PerformanceTier::Medium,
            _ => PerformanceTier::Low,
        }
    }

    /// Get model recommendations for this hardware
    pub fn recommend_models(&self, hardware: &HardwareInfo) -> Vec<ModelRecommendation> {
        let mut recommendations = Vec::new();

        // Known ternary models with their characteristics
        let known_models = [
            ("bitnet-b1.58-2b-4t", 2, 0.5),   // 2B params, ~0.5GB
            ("bitnet-b1.58-7b-4t", 7, 1.5),   // 7B params, ~1.5GB
            ("bitnet-b1.58-14b-4t", 14, 3.0), // 14B params, ~3GB
            ("bitnet-b1.58-32b-4t", 32, 7.0), // 32B params, ~7GB
        ];

        let available_ram_gb = hardware.memory.available_bytes as f32 / (1024.0 * 1024.0 * 1024.0);
        let base_perf = match hardware.performance_tier {
            PerformanceTier::Low => 10.0,
            PerformanceTier::Medium => 25.0,
            PerformanceTier::High => 50.0,
            PerformanceTier::Enthusiast => 100.0,
        };

        for (model_id, size_b, size_gb) in known_models {
            let fits = size_gb < available_ram_gb * 0.8; // Leave 20% headroom
            let estimated_tps = base_perf * (2.0 / size_b as f32).sqrt(); // Rough scaling
            let confidence = if fits { 0.9 } else { 0.3 };

            recommendations.push(ModelRecommendation {
                model_id: model_id.to_string(),
                size_b,
                estimated_ram_gb: size_gb,
                estimated_tokens_per_sec: estimated_tps,
                fits_in_memory: fits,
                confidence,
            });
        }

        // Sort by: fits in memory first, then by size (larger preferred if fits)
        recommendations.sort_by(|a, b| {
            b.fits_in_memory.cmp(&a.fits_in_memory)
                .then(b.size_b.cmp(&a.size_b))
        });

        recommendations
    }

    /// Estimate power consumption for inference
    pub fn estimate_power_watts(&self, hardware: &HardwareInfo, threads: usize, model_size_b: usize) -> f32 {
        // Very rough estimation based on CPU TDP scaling
        let base_tdp = match hardware.performance_tier {
            PerformanceTier::Low => 15.0,
            PerformanceTier::Medium => 45.0,
            PerformanceTier::High => 85.0,
            PerformanceTier::Enthusiast => 150.0,
        };

        let thread_factor = threads as f32 / hardware.cpu.physical_cores as f32;
        let model_factor = (model_size_b as f32 / 7.0).max(0.5);

        base_tdp * thread_factor * model_factor * 0.7 // 70% utilization estimate
    }
}

impl Default for HardwareDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let mut detector = HardwareDetector::new();
        let hw = detector.detect();

        assert!(!hw.cpu.brand.is_empty());
        assert!(hw.cpu.logical_cores > 0);
        assert!(hw.memory.total_bytes > 0);
        assert!(!hw.os.name.is_empty());
    }

    #[test]
    fn test_performance_tier_classification() {
        let detector = HardwareDetector::new();

        // Low tier
        let cpu = CpuInfo {
            physical_cores: 2,
            logical_cores: 4,
            ..Default::default()
        };
        let mem = MemoryInfo {
            total_bytes: 4 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(
            detector.classify_performance_tier(&cpu, &mem),
            PerformanceTier::Low
        );

        // High tier
        let cpu = CpuInfo {
            physical_cores: 12,
            logical_cores: 24,
            ..Default::default()
        };
        let mem = MemoryInfo {
            total_bytes: 32 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(
            detector.classify_performance_tier(&cpu, &mem),
            PerformanceTier::High
        );
    }
}

// Need Default for CpuInfo and MemoryInfo for tests
impl Default for CpuInfo {
    fn default() -> Self {
        Self {
            brand: String::new(),
            physical_cores: 0,
            logical_cores: 0,
            base_frequency_mhz: 0,
            max_frequency_mhz: 0,
            vendor_id: String::new(),
            l1_cache_kb: 0,
            l2_cache_kb: 0,
            l3_cache_kb: 0,
        }
    }
}

impl Default for MemoryInfo {
    fn default() -> Self {
        Self {
            total_bytes: 0,
            available_bytes: 0,
            used_bytes: 0,
            speed_mt_s: None,
        }
    }
}