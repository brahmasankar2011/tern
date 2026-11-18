//! Model management: loading, caching, and quantization

use crate::hardware::{HardwareInfo, ModelRecommendation};
use crate::tensor::{PackedTernaryTensor, TernaryLinear, TernaryQuantParams, TernaryWeight};
use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use ndarray::{Array, ArrayD, IxDyn};
use safetensors::tensor::TensorView;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique model identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Model architecture (e.g., "bitnet", "llama")
    pub architecture: String,
    /// Number of parameters (billions)
    pub params_b: f32,
    /// Quantization type
    pub quantization: String,
    /// Context length
    pub context_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Intermediate size (FFN)
    pub intermediate_size: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta
    pub rope_theta: f32,
    /// Model file path
    pub path: PathBuf,
    /// License
    pub license: String,
    /// Tags
    pub tags: Vec<String>,
}

/// Model configuration for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Inference settings
    pub inference: InferenceConfig,
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Number of threads
    pub threads: usize,
    /// Context length
    pub context_length: usize,
    /// Batch size
    pub batch_size: usize,
    /// Temperature
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Min-p sampling
    pub min_p: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Random seed
    pub seed: Option<u64>,
    /// Use flash attention if available
    pub flash_attention: bool,
    /// CPU affinity mask
    pub cpu_affinity: Option<Vec<usize>>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            threads: 0, // Auto-detect
            context_length: 4096,
            batch_size: 1,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            min_p: 0.0,
            repetition_penalty: 1.1,
            seed: None,
            flash_attention: false,
            cpu_affinity: None,
        }
    }
}

/// Loaded model with weights
#[derive(Debug)]
pub struct LoadedModel {
    pub metadata: ModelMetadata,
    pub config: InferenceConfig,
    pub embeddings: Arc<ArrayD<f32>>,
    pub layers: Vec<ModelLayer>,
    pub output_norm: Option<LayerNorm>,
    pub output_proj: Option<TernaryLinear>,
    pub tokenizer: Option<tokenizers::Tokenizer>,
}

/// Single transformer layer weights
#[derive(Debug)]
pub struct ModelLayer {
    pub attention_norm: LayerNorm,
    pub q_proj: TernaryLinear,
    pub k_proj: TernaryLinear,
    pub v_proj: TernaryLinear,
    pub o_proj: TernaryLinear,
    pub ffn_norm: LayerNorm,
    pub gate_proj: TernaryLinear,
    pub up_proj: TernaryLinear,
    pub down_proj: TernaryLinear,
}

/// Layer normalization weights
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub weight: ArrayD<f32>,
    pub bias: Option<ArrayD<f32>>,
    pub eps: f32,
}

/// Model loading error
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model not found: {0}")]
    NotFound(String),
    #[error("Invalid model format: {0}")]
    InvalidFormat(String),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Quantization error: {0}")]
    QuantizationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("HuggingFace Hub error: {0}")]
    HubError(#[from] hf_hub::api::sync::ApiError),
    #[error("Safetensors error: {0}")]
    SafetensorsError(#[from] safetensors::SafeTensorError),
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    #[error("Other: {0}")]
    Other(#[from] anyhow::Error),
}

/// Model manager for loading and caching models
pub struct ModelManager {
    /// Base directory for model storage
    cache_dir: PathBuf,
    /// Loaded models cache
    loaded_models: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,
    /// Hardware info for recommendations
    hardware: HardwareInfo,
    /// HF API client
    hf_api: Option<Api>,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new(cache_dir: PathBuf, hardware: HardwareInfo) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;

        let hf_api = Api::new().ok();

        Ok(Self {
            cache_dir,
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            hardware,
            hf_api,
        })
    }

    /// Get model recommendations for current hardware
    pub fn recommend_models(&self) -> Vec<ModelRecommendation> {
        let detector = crate::hardware::HardwareDetector::new();
        detector.recommend_models(&self.hardware)
    }

    /// Load a model by ID (from cache or download)
    pub async fn load_model(&self, model_id: &str, config: Option<InferenceConfig>) -> Result<Arc<LoadedModel>, ModelError> {
        // Check cache first
        {
            let cache = self.loaded_models.read().await;
            if let Some(model) = cache.get(model_id) {
                info!("Model {} loaded from cache", model_id);
                return Ok(model.clone());
            }
        }

        // Find model path
        let model_path = self.resolve_model_path(model_id).await?;

        // Load model from disk
        let model = self.load_model_from_path(&model_path, config).await?;

        // Cache it
        let model_arc = Arc::new(model);
        {
            let mut cache = self.loaded_models.write().await;
            cache.insert(model_id.to_string(), model_arc.clone());
        }

        info!("Model {} loaded and cached", model_id);
        Ok(model_arc)
    }

    /// Resolve model path (local or download from HF)
    async fn resolve_model_path(&self, model_id: &str) -> Result<PathBuf, ModelError> {
        // Check local cache first
        let local_path = self.cache_dir.join(model_id);
        if local_path.exists() {
            return Ok(local_path);
        }

        // Try to download from Hugging Face
        if let Some(api) = &self.hf_api {
            info!("Downloading model {} from Hugging Face", model_id);
            let repo = Repo::new(model_id.to_string(), RepoType::Model);
            let api = api.repo(repo);

            // Download all files
            let local_dir = api
                .download_dir()
                .context("Failed to download model")?;

            return Ok(local_dir);
        }

        Err(ModelError::NotFound(format!(
            "Model {} not found locally and no HF API configured",
            model_id
        )))
    }

    /// Load model from a local path
    async fn load_model_from_path(
        &self,
        path: &Path,
        config: Option<InferenceConfig>,
    ) -> Result<LoadedModel, ModelError> {
        info!("Loading model from {}", path.display());

        // Find model files
        let model_files = self.find_model_files(path)?;
        let config_file = model_files
            .iter()
            .find(|f| f.extension().map_or(false, |ext| ext == "json"))
            .context("No config.json found")?;

        let weight_files: Vec<_> = model_files
            .iter()
            .filter(|f| f.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();

        // Load config
        let config_json = std::fs::read_to_string(config_file)
            .context("Failed to read config.json")?;
        let metadata: ModelMetadata = serde_json::from_str(&config_json)
            .context("Failed to parse config.json")?;

        // Load weights
        let weights = self.load_weights(&weight_files)?;

        // Load tokenizer if present
        let tokenizer = self.load_tokenizer(path)?;

        // Build model structure
        let model = self.build_model(metadata, weights, tokenizer, config)?;

        Ok(model)
    }

    /// Find model files in directory
    fn find_model_files(&self, path: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            }
        }
        Ok(files)
    }

    /// Load weights from safetensors files
    fn load_weights(&self, files: &[&PathBuf]) -> Result<HashMap<String, ArrayD<f32>>> {
        let mut weights = HashMap::new();

        for file in files {
            let data = std::fs::read(file)?;
            let tensors = safetensors::SafeTensors::deserialize(&data)?;

            for name in tensors.names() {
                let tensor: TensorView = tensors.tensor(&name)?;
                let shape: Vec<usize> = tensor.shape().iter().map(|&d| d as usize).collect();
                let data: Vec<f32> = tensor
                    .data()
                    .chunks(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .context(format!("Failed to create array for {}", name))?;
                weights.insert(name.to_string(), array);
            }
        }

        Ok(weights)
    }

    /// Load tokenizer
    fn load_tokenizer(&self, path: &Path) -> Result<Option<tokenizers::Tokenizer>> {
        let tokenizer_path = path.join("tokenizer.json");
        if tokenizer_path.exists() {
            let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
                .map_err(|e| ModelError::TokenizerError(e.to_string()))?;
            Ok(Some(tokenizer))
        } else {
            Ok(None)
        }
    }

    /// Build model from weights
    fn build_model(
        &self,
        metadata: ModelMetadata,
        weights: HashMap<String, ArrayD<f32>>,
        tokenizer: Option<tokenizers::Tokenizer>,
        inference_config: Option<InferenceConfig>,
    ) -> Result<LoadedModel, ModelError> {
        let config = inference_config.unwrap_or_default();

        // Extract embeddings
        let embeddings = weights
            .get("model.embed_tokens.weight")
            .or_else(|| weights.get("embed_tokens.weight"))
            .context("Missing embeddings weight")?
            .clone();

        // Build layers
        let mut layers = Vec::new();
        for layer_idx in 0..metadata.num_layers {
            let prefix = format!("model.layers.{}.", layer_idx);

            let layer = ModelLayer {
                attention_norm: self.load_layer_norm(&weights, &format!("{}input_layernorm", prefix))?,
                q_proj: self.load_ternary_linear(&weights, &format!("{}self_attn.q_proj", prefix), metadata.hidden_size, metadata.hidden_size)?,
                k_proj: self.load_ternary_linear(&weights, &format!("{}self_attn.k_proj", prefix), metadata.hidden_size, metadata.num_kv_heads * (metadata.hidden_size / metadata.num_heads))?,
                v_proj: self.load_ternary_linear(&weights, &format!("{}self_attn.v_proj", prefix), metadata.hidden_size, metadata.num_kv_heads * (metadata.hidden_size / metadata.num_heads))?,
                o_proj: self.load_ternary_linear(&weights, &format!("{}self_attn.o_proj", prefix), metadata.hidden_size, metadata.hidden_size)?,
                ffn_norm: self.load_layer_norm(&weights, &format!("{}post_attention_layernorm", prefix))?,
                gate_proj: self.load_ternary_linear(&weights, &format!("{}mlp.gate_proj", prefix), metadata.hidden_size, metadata.intermediate_size)?,
                up_proj: self.load_ternary_linear(&weights, &format!("{}mlp.up_proj", prefix), metadata.hidden_size, metadata.intermediate_size)?,
                down_proj: self.load_ternary_linear(&weights, &format!("{}mlp.down_proj", prefix), metadata.intermediate_size, metadata.hidden_size)?,
            };
            layers.push(layer);
        }

        // Output norm
        let output_norm = self.load_layer_norm(&weights, "model.norm").ok();

        // Output projection (tied to embeddings usually)
        let output_proj = if let Some(w) = weights.get("lm_head.weight") {
            Some(self.convert_to_ternary_linear(w.clone(), metadata.hidden_size, metadata.vocab_size)?)
        } else {
            None
        };

        Ok(LoadedModel {
            metadata,
            config,
            embeddings: Arc::new(embeddings),
            layers,
            output_norm,
            output_proj,
            tokenizer,
        })
    }

    /// Load layer norm from weights
    fn load_layer_norm(&self, weights: &HashMap<String, ArrayD<f32>>, prefix: &str) -> Result<LayerNorm> {
        let weight = weights
            .get(&format!("{}.weight", prefix))
            .context(format!("Missing {} weight", prefix))?
            .clone();

        let bias = weights.get(&format!("{}.bias", prefix)).cloned();

        Ok(LayerNorm {
            weight,
            bias,
            eps: 1e-6,
        })
    }

    /// Load ternary linear layer from weights
    fn load_ternary_linear(
        &self,
        weights: &HashMap<String, ArrayD<f32>>,
        prefix: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<TernaryLinear> {
        let weight = weights
            .get(&format!("{}.weight", prefix))
            .context(format!("Missing {} weight", prefix))?;

        // Convert to ternary
        let ternary_weights = self.quantize_to_ternary(weight.view(), out_features, in_features)?;

        let bias = weights.get(&format!("{}.bias", prefix)).cloned();

        TernaryLinear::new(
            ternary_weights,
            bias,
            TernaryQuantParams::default(),
        ).map_err(|e| ModelError::QuantizationError(e.to_string()))
    }

    /// Convert full precision weight to ternary linear
    fn convert_to_ternary_linear(
        &self,
        weight: ArrayD<f32>,
        in_features: usize,
        out_features: usize,
    ) -> Result<TernaryLinear> {
        let weight_view = weight.view();
        let ternary_weights = self.quantize_to_ternary(weight_view, out_features, in_features)?;

        TernaryLinear::new(
            ternary_weights,
            None,
            TernaryQuantParams::default(),
        ).map_err(|e| ModelError::QuantizationError(e.to_string()))
    }

    /// Quantize weights to ternary
    fn quantize_to_ternary(
        &self,
        weight: ndarray::ArrayView<f32, IxDyn>,
        out_features: usize,
        in_features: usize,
    ) -> Result<PackedTernaryTensor> {
        let shape = weight.shape();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], out_features);
        assert_eq!(shape[1], in_features);

        let threshold = 0.02f32; // Standard ternary threshold
        let mut ternary_weights = Vec::with_capacity(out_features * in_features);

        for out_idx in 0..out_features {
            for in_idx in 0..in_features {
                let val = weight[[out_idx, in_idx]];
                ternary_weights.push(TernaryWeight::from_f32(val, threshold));
            }
        }

        Ok(PackedTernaryTensor::new(&ternary_weights, vec![out_features, in_features]))
    }

    /// Unload a model from cache
    pub async fn unload_model(&self, model_id: &str) -> bool {
        let mut cache = self.loaded_models.write().await;
        cache.remove(model_id).is_some()
    }

    /// List loaded models
    pub async fn list_loaded_models(&self) -> Vec<String> {
        let cache = self.loaded_models.read().await;
        cache.keys().cloned().collect()
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            threads: 0,
            context_length: 4096,
            batch_size: 1,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            min_p: 0.0,
            repetition_penalty: 1.1,
            seed: None,
            flash_attention: false,
            cpu_affinity: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let detector = crate::hardware::HardwareDetector::new();
        let hw = detector.detect();

        let manager = ModelManager::new(temp_dir.path().to_path_buf(), hw).unwrap();
        assert_eq!(manager.cache_dir(), temp_dir.path());
    }
}