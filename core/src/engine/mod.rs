//! Engine interface and inference implementation

use crate::model::{InferenceConfig, LoadedModel, ModelError, ModelMetadata};
use crate::tensor::{activation, layer_norm, PackedTernaryTensor, TernaryLinear, TernaryWeight, TensorError};
use anyhow::{Context, Result};
use ndarray::{Array, ArrayD, ArrayView, Axis, IxDyn, s};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, trace};

/// Engine trait for pluggable inference backends
#[async_trait::async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Engine name
    fn name(&self) -> &'static str;

    /// Engine version
    fn version(&self) -> &'static str;

    /// Supported model architectures
    fn supported_architectures(&self) -> Vec<&'static str>;

    /// Initialize engine with model
    async fn initialize(&mut self, model: Arc<LoadedModel>) -> Result<(), EngineError>;

    /// Run inference
    async fn infer(&self, input: &InferenceInput) -> Result<InferenceOutput, EngineError>;

    /// Run streaming inference
    async fn infer_stream(
        &self,
        input: &InferenceInput,
        callback: Box<dyn FnMut(&InferenceChunk) -> Result<()> + Send>,
    ) -> Result<(), EngineError>;

    /// Get engine stats
    fn stats(&self) -> EngineStats;

    /// Shutdown engine
    async fn shutdown(&mut self) -> Result<(), EngineError>;
}

/// Inference input
#[derive(Debug, Clone)]
pub struct InferenceInput {
    /// Input token IDs
    pub tokens: Vec<u32>,
    /// Inference configuration
    pub config: InferenceConfig,
    /// Previous KV cache (for continuation)
    pub kv_cache: Option<KVCache>,
}

/// Inference output
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    /// Next token logits [vocab_size]
    pub logits: ArrayD<f32>,
    /// Updated KV cache
    pub kv_cache: KVCache,
    /// Number of tokens processed
    pub tokens_processed: usize,
}

/// Streaming inference chunk
#[derive(Debug, Clone)]
pub struct InferenceChunk {
    /// Token ID (if complete)
    pub token: Option<u32>,
    /// Logits for next token
    pub logits: Option<ArrayD<f32>>,
    /// Is this the final chunk
    pub is_final: bool,
}

/// Key-Value cache for attention
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Per-layer key cache [num_layers, 2, batch, seq_len, num_heads, head_dim]
    pub keys: Vec<ArrayD<f32>>,
    /// Per-layer value cache [num_layers, 2, batch, seq_len, num_heads, head_dim]
    pub values: Vec<ArrayD<f32>>,
}

impl KVCache {
    /// Create empty KV cache for model
    pub fn new(num_layers: usize, batch_size: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let shape = IxDyn(&[batch_size, max_seq_len, num_heads, head_dim]);
        Self {
            keys: vec![ArrayD::zeros(shape.clone()); num_layers],
            values: vec![ArrayD::zeros(shape); num_layers],
        }
    }

    /// Get cache for a specific layer
    pub fn layer(&self, layer_idx: usize) -> (&ArrayD<f32>, &ArrayD<f32>) {
        (&self.keys[layer_idx], &self.values[layer_idx])
    }

    /// Get mutable cache for a specific layer
    pub fn layer_mut(&mut self, layer_idx: usize) -> (&mut ArrayD<f32>, &mut ArrayD<f32>) {
        (&mut self.keys[layer_idx], &mut self.values[layer_idx])
    }

    /// Append new key/value to cache
    pub fn append(&mut self, layer_idx: usize, seq_idx: usize, key: &ArrayView<f32, IxDyn>, value: &ArrayView<f32, IxDyn>) -> Result<(), TensorError> {
        let key_cache = &mut self.keys[layer_idx];
        let value_cache = &mut self.values[layer_idx];

        let key_slice = key_cache.slice_mut(s![.., seq_idx, .., ..]);
        let value_slice = value_cache.slice_mut(s![.., seq_idx, .., ..]);

        key_slice.assign(key);
        value_slice.assign(value);

        Ok(())
    }
}

/// Engine statistics
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    /// Total tokens generated
    pub total_tokens: u64,
    /// Total inference time (ms)
    pub total_time_ms: u64,
    /// Average tokens per second
    pub avg_tokens_per_sec: f32,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: u64,
    /// Current memory usage (bytes)
    pub current_memory_bytes: u64,
}

/// Engine errors
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Engine not initialized")]
    NotInitialized,
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),
    #[error("Model error: {0}")]
    ModelError(#[from] ModelError),
    #[error("Other: {0}")]
    Other(#[from] anyhow::Error),
}

/// Built-in ternary inference engine
pub struct TernaryEngine {
    model: Option<Arc<LoadedModel>>,
    config: InferenceConfig,
    rng: Mutex<StdRng>,
    stats: Mutex<EngineStats>,
    kv_cache: Mutex<Option<KVCache>>,
}

impl TernaryEngine {
    /// Create a new ternary engine
    pub fn new(config: InferenceConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| rand::random());
        Self {
            model: None,
            config,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
            stats: Mutex::new(EngineStats::default()),
            kv_cache: Mutex::new(None),
        }
    }

    /// Get model reference
    pub fn model(&self) -> Option<&Arc<LoadedModel>> {
        self.model.as_ref()
    }

    /// Apply RoPE (Rotary Positional Embedding)
    fn apply_rope(
        &self,
        x: &mut ArrayD<f32>,
        position_ids: &[usize],
        head_dim: usize,
        rope_theta: f32,
    ) -> Result<(), TensorError> {
        let shape = x.shape().to_vec();
        let (batch, seq_len, num_heads, _) = (shape[0], shape[1], shape[2], shape[3]);

        // Precompute frequencies
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        for b in 0..batch {
            for s in 0..seq_len {
                let pos = position_ids.get(s).copied().unwrap_or(s);
                for h in 0..num_heads {
                    for i in 0..head_dim / 2 {
                        let freq = inv_freq[i];
                        let theta = pos as f32 * freq;
                        let cos = theta.cos();
                        let sin = theta.sin();

                        let idx1 = [[b, s, h, 2 * i]];
                        let idx2 = [[b, s, h, 2 * i + 1]];

                        let x1 = x[idx1];
                        let x2 = x[idx2];

                        x[idx1] = x1 * cos - x2 * sin;
                        x[idx2] = x1 * sin + x2 * cos;
                    }
                }
            }
        }

        Ok(())
    }

    /// Scaled dot-product attention
    fn attention(
        &self,
        q: &ArrayD<f32>,
        k: &ArrayD<f32>,
        v: &ArrayD<f32>,
        mask: Option<&ArrayD<f32>>,
    ) -> Result<ArrayD<f32>, TensorError> {
        // q: [batch, seq_len, num_heads, head_dim]
        // k: [batch, seq_len, num_kv_heads, head_dim]
        // v: [batch, seq_len, num_kv_heads, head_dim]

        let q_shape = q.shape();
        let batch = q_shape[0];
        let seq_len_q = q_shape[1];
        let num_heads = q_shape[2];
        let head_dim = q_shape[3];

        let k_shape = k.shape();
        let seq_len_k = k_shape[1];
        let num_kv_heads = k_shape[2];

        // Repeat K/V for GQA
        let repeat_factor = num_heads / num_kv_heads;

        // Compute attention scores: Q @ K^T / sqrt(d)
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut output = ArrayD::zeros(IxDyn(&[batch, seq_len_q, num_heads, head_dim]));

        for b in 0..batch {
            for h in 0..num_heads {
                let kv_h = h / repeat_factor;

                for sq in 0..seq_len_q {
                    // Compute scores for this query position
                    let mut scores = vec![0.0f32; seq_len_k];

                    for sk in 0..seq_len_k {
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q[[b, sq, h, d]] * k[[b, sk, kv_h, d]];
                        }
                        scores[sk] = score * scale;
                    }

                    // Apply mask if provided
                    if let Some(mask) = mask {
                        for sk in 0..seq_len_k {
                            scores[sk] += mask[[b, 0, sq, sk]];
                        }
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
                    for sk in 0..seq_len_k {
                        scores[sk] = (scores[sk] - max_score).exp() / exp_sum;
                    }

                    // Weighted sum of values
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for sk in 0..seq_len_k {
                            acc += scores[sk] * v[[b, sk, kv_h, d]];
                        }
                        output[[b, sq, h, d]] = acc;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Forward pass through a single transformer layer
    fn forward_layer(
        &self,
        layer: &crate::model::ModelLayer,
        hidden_states: &mut ArrayD<f32>,
        position_ids: &[usize],
        kv_cache: &mut KVCache,
        layer_idx: usize,
        metadata: &ModelMetadata,
    ) -> Result<(), TensorError> {
        let residual = hidden_states.clone();

        // Attention norm
        let mut normed = hidden_states.clone();
        layer_norm::LayerNorm::forward(&layer.attention_norm, &mut normed);

        // Project Q, K, V
        let q = layer.q_proj.forward(&normed)?;
        let k = layer.k_proj.forward(&normed)?;
        let v = layer.v_proj.forward(&normed)?;

        // Reshape for attention: [batch, seq, heads, head_dim]
        let batch = q.shape()[0];
        let seq_len = q.shape()[1];
        let head_dim = metadata.hidden_size / metadata.num_heads;

        let mut q = q.into_shape(IxDyn(&[batch, seq_len, metadata.num_heads, head_dim]))?;
        let mut k = k.into_shape(IxDyn(&[batch, seq_len, metadata.num_kv_heads, head_dim]))?;
        let mut v = v.into_shape(IxDyn(&[batch, seq_len, metadata.num_kv_heads, head_dim]))?;

        // Apply RoPE
        self.apply_rope(&mut q, position_ids, head_dim, metadata.rope_theta)?;
        self.apply_rope(&mut k, position_ids, head_dim, metadata.rope_theta)?;

        // Update KV cache
        let cache_seq_len = kv_cache.keys[layer_idx].shape()[1];
        let start_pos = cache_seq_len - seq_len; // Simplified

        for b in 0..batch {
            for s in 0..seq_len {
                let cache_pos = start_pos + s;
                if cache_pos < cache_seq_len {
                    kv_cache.append(layer_idx, cache_pos, &q.slice(s![b, s, .., ..]), &v.slice(s![b, s, .., ..]))?;
                }
            }
        }

        // Get full K/V from cache
        let cached_k = kv_cache.keys[layer_idx].clone();
        let cached_v = kv_cache.values[layer_idx].clone();

        // Attention
        let causal_mask = self.create_causal_mask(batch, seq_len, cache_seq_len)?;
        let attn_output = self.attention(&q, &cached_k, &cached_v, Some(&causal_mask))?;

        // Reshape back
        let attn_output = attn_output.into_shape(IxDyn(&[batch, seq_len, metadata.hidden_size]))?;

        // Output projection
        let attn_output = layer.o_proj.forward(&attn_output)?;

        // Residual
        *hidden_states += &attn_output;

        // FFN
        let residual = hidden_states.clone();
        let mut normed = hidden_states.clone();
        layer_norm::LayerNorm::forward(&layer.ffn_norm, &mut normed);

        let gate = layer.gate_proj.forward(&normed)?;
        let up = layer.up_proj.forward(&normed)?;

        // SiLU activation on gate
        activation::silu(&mut gate);

        // Element-wise multiply
        let mut ffn = gate * up;

        let ffn = layer.down_proj.forward(&ffn)?;

        *hidden_states = residual + ffn;

        Ok(())
    }

    /// Create causal attention mask
    fn create_causal_mask(
        &self,
        batch: usize,
        seq_len: usize,
        cache_len: usize,
    ) -> Result<ArrayD<f32>, TensorError> {
        let mut mask = ArrayD::zeros(IxDyn(&[batch, 1, seq_len, cache_len]));

        for b in 0..batch {
            for i in 0..seq_len {
                for j in 0..cache_len {
                    if j > (cache_len - seq_len) + i {
                        mask[[b, 0, i, j]] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        Ok(mask)
    }

    /// Sample next token from logits
    fn sample_token(
        &self,
        logits: &ArrayD<f32>,
        config: &InferenceConfig,
        rng: &mut StdRng,
    ) -> u32 {
        let vocab_size = logits.shape()[logits.shape().len() - 1];
        let logits_slice = logits.slice(s![.., -1, ..]); // Last token

        // Apply temperature
        let mut probs: Vec<f32> = logits_slice
            .iter()
            .map(|&x| (x / config.temperature).exp())
            .collect();

        // Apply top-k
        if config.top_k > 0 && config.top_k < vocab_size {
            let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let threshold = indexed[config.top_k - 1].1;
            for (i, p) in probs.iter_mut().enumerate() {
                if *p < threshold {
                    *p = 0.0;
                }
            }
        }

        // Apply top-p
        if config.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut cumsum = 0.0;
            let mut cutoff = vocab_size;
            for (i, (idx, p)) in indexed.iter().enumerate() {
                cumsum += *p;
                if cumsum >= config.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            for (i, (idx, _)) in indexed.iter().enumerate() {
                if i >= cutoff {
                    probs[*idx] = 0.0;
                }
            }
        }

        // Apply min-p
        if config.min_p > 0.0 {
            let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
            let threshold = config.min_p * max_prob;
            for p in &mut probs {
                if *p < threshold {
                    *p = 0.0;
                }
            }
        }

        // Apply repetition penalty (simplified - would need token history)
        // ...

        // Normalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            // Fallback: uniform
            for p in &mut probs {
                *p = 1.0 / vocab_size as f32;
            }
        }

        // Sample
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return i as u32;
            }
        }

        (vocab_size - 1) as u32
    }
}

#[async_trait::async_trait]
impl InferenceEngine for TernaryEngine {
    fn name(&self) -> &'static str {
        "ternary-engine"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn supported_architectures(&self) -> Vec<&'static str> {
        vec!["bitnet", "llama", "mistral"]
    }

    async fn initialize(&mut self, model: Arc<LoadedModel>) -> Result<(), EngineError> {
        info!("Initializing ternary engine with model: {}", model.metadata.id);

        // Initialize KV cache
        let metadata = &model.metadata;
        let batch_size = self.config.batch_size;
        let max_seq_len = self.config.context_length;
        let num_heads = metadata.num_heads;
        let head_dim = metadata.hidden_size / metadata.num_heads;

        let kv_cache = KVCache::new(
            metadata.num_layers,
            batch_size,
            num_heads,
            head_dim,
            max_seq_len,
        );

        *self.kv_cache.lock().await = Some(kv_cache);
        self.model = Some(model);

        Ok(())
    }

    async fn infer(&self, input: &InferenceInput) -> Result<InferenceOutput, EngineError> {
        let model = self.model.as_ref().ok_or(EngineError::ModelNotLoaded)?;
        let metadata = &model.metadata;

        let start_time = std::time::Instant::now();

        // Get or create KV cache
        let mut kv_cache = input.kv_cache.clone().unwrap_or_else(|| {
            KVCache::new(
                metadata.num_layers,
                input.config.batch_size,
                metadata.num_heads,
                metadata.hidden_size / metadata.num_heads,
                input.config.context_length,
            )
        });

        // Embed tokens
        let tokens = ArrayD::from_shape_vec(
            IxDyn(&[input.config.batch_size, input.tokens.len()]),
            input.tokens.clone(),
        )?;

        let mut hidden_states = model.embeddings.as_ref().forward(&tokens)?;

        // Position IDs
        let position_ids: Vec<usize> = (0..input.tokens.len()).collect();

        // Forward through layers
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            self.forward_layer(layer, &mut hidden_states, &position_ids, &mut kv_cache, layer_idx, metadata)?;
        }

        // Output norm
        if let Some(norm) = &model.output_norm {
            layer_norm::LayerNorm::forward(norm, &mut hidden_states);
        }

        // Output projection
        let logits = if let Some(proj) = &model.output_proj {
            proj.forward(&hidden_states)?
        } else {
            // Tied embeddings
            let vocab_size = metadata.vocab_size;
            let hidden_size = metadata.hidden_size;
            let batch_seq = hidden_states.shape()[0] * hidden_states.shape()[1];
            let hidden_2d = hidden_states.into_shape((batch_seq, hidden_size))?;
            let emb_t = model.embeddings.t();
            let logits_2d = hidden_2d.dot(&emb_t);
            logits_2d.into_shape(IxDyn(&[input.config.batch_size, input.tokens.len(), vocab_size]))?
        };

        // Update stats
        let elapsed = start_time.elapsed();
        {
            let mut stats = self.stats.lock().await;
            stats.total_tokens += input.tokens.len() as u64;
            stats.total_time_ms += elapsed.as_millis() as u64;
            stats.avg_tokens_per_sec = stats.total_tokens as f32 / (stats.total_time_ms as f32 / 1000.0);
        }

        // Return logits for last token only
        let last_logits = logits.slice(s![.., -1, ..]).to_owned();

        Ok(InferenceOutput {
            logits: last_logits,
            kv_cache,
            tokens_processed: input.tokens.len(),
        })
    }

    async fn infer_stream(
        &self,
        input: &InferenceInput,
        mut callback: Box<dyn FnMut(&InferenceChunk) -> Result<()> + Send>,
    ) -> Result<(), EngineError> {
        let mut current_input = input.clone();

        loop {
            let output = self.infer(&current_input).await?;

            // Sample next token
            let token = {
                let mut rng = self.rng.lock().await;
                self.sample_token(&output.logits, &input.config, &mut rng)
            };

            // Send chunk
            let is_final = token == 0 || token == 2; // EOS tokens (model-dependent)
            callback(&InferenceChunk {
                token: Some(token),
                logits: Some(output.logits.clone()),
                is_final,
            })?;

            if is_final {
                break;
            }

            // Continue with new token
            current_input.tokens = vec![token];
            current_input.kv_cache = Some(output.kv_cache);
        }

        Ok(())
    }

    fn stats(&self) -> EngineStats {
        // This is a blocking call in async trait - in practice use try_lock or restructure
        EngineStats::default()
    }

    async fn shutdown(&mut self) -> Result<(), EngineError> {
        info!("Shutting down ternary engine");
        self.model = None;
        *self.kv_cache.lock().await = None;
        Ok(())
    }
}

/// Engine factory for creating engines
pub struct EngineFactory;

impl EngineFactory {
    /// Create an engine by name
    pub fn create(name: &str, config: InferenceConfig) -> Result<Box<dyn InferenceEngine>, EngineError> {
        match name {
            "ternary" | "ternary-engine" => Ok(Box::new(TernaryEngine::new(config))),
            "bitnet" => {
                // Would create bitnet.cpp engine
                Err(EngineError::InvalidInput("bitnet engine not implemented yet".into()))
            }
            "llamacpp" => {
                // Would create llama.cpp engine
                Err(EngineError::InvalidInput("llamacpp engine not implemented yet".into()))
            }
            _ => Err(EngineError::InvalidInput(format!("Unknown engine: {}", name))),
        }
    }

    /// List available engines
    pub fn available_engines() -> Vec<&'static str> {
        vec!["ternary", "bitnet", "llamacpp"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::InferenceConfig;

    #[test]
    fn test_engine_factory() {
        let config = InferenceConfig::default();
        let engine = EngineFactory::create("ternary", config).unwrap();
        assert_eq!(engine.name(), "ternary-engine");
    }

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new(32, 1, 32, 128, 4096);
        assert_eq!(cache.keys.len(), 32);
        assert_eq!(cache.values.len(), 32);
    }

    #[test]
    fn test_causal_mask() {
        let engine = TernaryEngine::new(InferenceConfig::default());
        let mask = engine.create_causal_mask(1, 4, 4).unwrap();
        assert_eq!(mask.shape(), &[1, 1, 4, 4]);

        // Check causal property: upper triangle should be -inf
        for i in 0..4 {
            for j in 0..4 {
                if j > i {
                    assert_eq!(mask[[0, 0, i, j]], f32::NEG_INFINITY);
                } else {
                    assert_eq!(mask[[0, 0, i, j]], 0.0);
                }
            }
        }
    }
}