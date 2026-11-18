//! Tensor operations for ternary (1.58-bit) inference
//!
//! This module provides efficient tensor operations optimized for ternary weights
//! where values are -1, 0, or +1. Matrix multiplication becomes additions and lookups.

use ndarray::{Array, ArrayD, ArrayView, Dimension, IxDyn, s};
use num_traits::{Float, NumCast, Zero};
use std::fmt::Debug;

/// Ternary weight representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernaryWeight {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl TernaryWeight {
    /// Convert from f32 to ternary using threshold
    pub fn from_f32(value: f32, threshold: f32) -> Self {
        if value > threshold {
            TernaryWeight::Pos
        } else if value < -threshold {
            TernaryWeight::Neg
        } else {
            TernaryWeight::Zero
        }
    }

    /// Convert to f32
    pub fn to_f32(self) -> f32 {
        match self {
            TernaryWeight::Neg => -1.0,
            TernaryWeight::Zero => 0.0,
            TernaryWeight::Pos => 1.0,
        }
    }

    /// Convert to i8
    pub fn to_i8(self) -> i8 {
        match self {
            TernaryWeight::Neg => -1,
            TernaryWeight::Zero => 0,
            TernaryWeight::Pos => 1,
        }
    }
}

/// Packed ternary tensor for efficient storage
/// Uses 2 bits per weight: 00=Zero, 01=Pos, 10=Neg (11 unused)
#[derive(Debug, Clone)]
pub struct PackedTernaryTensor {
    data: Vec<u8>,
    shape: Vec<usize>,
}

impl PackedTernaryTensor {
    /// Create a new packed tensor from ternary weights
    pub fn new(weights: &[TernaryWeight], shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        assert_eq!(weights.len(), total_elements);

        let packed_len = (total_elements + 3) / 4;
        let mut data = vec![0u8; packed_len];

        for (i, &weight) in weights.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let value = match weight {
                TernaryWeight::Neg => 0b10,
                TernaryWeight::Zero => 0b00,
                TernaryWeight::Pos => 0b01,
            };
            data[byte_idx] |= value << bit_offset;
        }

        Self { data, shape }
    }

    /// Get the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Unpack a single weight
    pub fn get(&self, index: usize) -> Option<TernaryWeight> {
        if index >= self.len() {
            return None;
        }
        let byte_idx = index / 4;
        let bit_offset = (index % 4) * 2;
        let value = (self.data[byte_idx] >> bit_offset) & 0b11;
        Some(match value {
            0b10 => TernaryWeight::Neg,
            0b01 => TernaryWeight::Pos,
            _ => TernaryWeight::Zero,
        })
    }

    /// Unpack all weights to a vector
    pub fn unpack(&self) -> Vec<TernaryWeight> {
        (0..self.len()).map(|i| self.get(i).unwrap()).collect()
    }

    /// Get raw packed data
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Create from raw bytes (for loading from disk)
    pub fn from_bytes(data: Vec<u8>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

/// Quantization parameters for ternary conversion
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TernaryQuantParams {
    /// Threshold for ternary quantization
    pub threshold: f32,
    /// Scale factor for dequantization
    pub scale: f32,
}

impl Default for TernaryQuantParams {
    fn default() -> Self {
        Self {
            threshold: 0.02,
            scale: 1.0,
        }
    }
}

/// Ternary linear layer (quantized weight matrix)
#[derive(Debug, Clone)]
pub struct TernaryLinear {
    /// Packed ternary weights [out_features, in_features]
    pub weight: PackedTernaryTensor,
    /// Optional bias [out_features]
    pub bias: Option<ArrayD<f32>>,
    /// Quantization parameters
    pub quant_params: TernaryQuantParams,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
}

impl TernaryLinear {
    /// Create a new ternary linear layer
    pub fn new(
        weight: PackedTernaryTensor,
        bias: Option<ArrayD<f32>>,
        quant_params: TernaryQuantParams,
    ) -> Result<Self, TensorError> {
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(TensorError::InvalidShape(
                "Weight must be 2D [out_features, in_features]".into(),
            ));
        }
        let out_features = shape[0];
        let in_features = shape[1];

        if let Some(ref b) = bias {
            if b.shape()[0] != out_features {
                return Err(TensorError::InvalidShape(
                    "Bias shape must match out_features".into(),
                ));
            }
        }

        Ok(Self {
            weight,
            bias,
            quant_params,
            in_features,
            out_features,
        })
    }

    /// Forward pass with ternary matrix multiplication
    /// Uses efficient addition-based computation for ternary weights
    pub fn forward(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>, TensorError> {
        // Input shape: [..., in_features]
        // Weight shape: [out_features, in_features]
        // Output shape: [..., out_features]

        let input_shape = input.shape();
        let in_features = input_shape[input_shape.len() - 1];
        if in_features != self.in_features {
            return Err(TensorError::InvalidShape(format!(
                "Input features {} != layer in_features {}",
                in_features, self.in_features
            )));
        }

        // Flatten all but last dimension
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let input_2d = input
            .view()
            .into_shape((batch_size, self.in_features))
            .map_err(|_| TensorError::InvalidShape("Failed to reshape input".into()))?;

        // Output buffer
        let mut output = ArrayD::zeros(IxDyn(&[batch_size, self.out_features]));

        // Ternary matrix multiplication: for each output feature
        for out_idx in 0..self.out_features {
            // Get weights for this output neuron
            let weight_start = out_idx * self.in_features;
            let weight_end = weight_start + self.in_features;

            // Accumulate dot products using ternary arithmetic
            for batch_idx in 0..batch_size {
                let mut acc = 0.0f32;
                for in_idx in 0..self.in_features {
                    let weight_idx = weight_start + in_idx;
                    if let Some(weight) = self.weight.get(weight_idx) {
                        match weight {
                            TernaryWeight::Pos => acc += input_2d[[batch_idx, in_idx]],
                            TernaryWeight::Neg => acc -= input_2d[[batch_idx, in_idx]],
                            TernaryWeight::Zero => {}
                        }
                    }
                }
                // Apply scale
                acc *= self.quant_params.scale;
                // Add bias if present
                if let Some(ref bias) = self.bias {
                    acc += bias[out_idx];
                }
                output[[batch_idx, out_idx]] = acc;
            }
        }

        // Restore original batch dimensions
        let mut out_shape = input_shape[..input_shape.len() - 1].to_vec();
        out_shape.push(self.out_features);
        output
            .into_shape(IxDyn(&out_shape))
            .map_err(|_| TensorError::InvalidShape("Failed to reshape output".into()))
    }
}

/// Tensor operations error
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("Invalid shape: {0}")]
    InvalidShape(String),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Quantization error: {0}")]
    QuantizationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Activation functions
pub mod activation {
    use ndarray::{Array, ArrayD, IxDyn};

    /// ReLU activation
    pub fn relu(x: &mut ArrayD<f32>) {
        x.mapv_inplace(|v| v.max(0.0));
    }

    /// GELU activation (approximate)
    pub fn gelu(x: &mut ArrayD<f32>) {
        const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
        const COEF: f32 = 0.044715;
        x.mapv_inplace(|v| {
            let x3 = v * v * v;
            let tanh_arg = SQRT_2_OVER_PI * (v + COEF * x3);
            0.5 * v * (1.0 + tanh_arg.tanh())
        });
    }

    /// SiLU (Swish) activation
    pub fn silu(x: &mut ArrayD<f32>) {
        x.mapv_inplace(|v| v / (1.0 + (-v).exp()));
    }

    /// Softmax along last dimension
    pub fn softmax(x: &mut ArrayD<f32>) {
        let shape = x.shape().to_vec();
        let last_dim = shape[shape.len() - 1];
        let batch_size: usize = shape[..shape.len() - 1].iter().product();

        let mut x_2d = x
            .view_mut()
            .into_shape((batch_size, last_dim))
            .unwrap();

        for mut row in x_2d.rows_mut() {
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|&v| (v - max_val).exp()).sum();
            row.mapv_inplace(|v| (v - max_val).exp() / exp_sum);
        }
    }
}

/// Layer normalization
pub mod layer_norm {
    use ndarray::{Array, ArrayD, IxDyn};

    #[derive(Debug, Clone)]
    pub struct LayerNorm {
        pub weight: ArrayD<f32>,
        pub bias: ArrayD<f32>,
        pub eps: f32,
    }

    impl LayerNorm {
        pub fn new(normalized_shape: usize, eps: f32) -> Self {
            Self {
                weight: ArrayD::ones(IxDyn(&[normalized_shape])),
                bias: ArrayD::zeros(IxDyn(&[normalized_shape])),
                eps,
            }
        }

        pub fn forward(&self, input: &mut ArrayD<f32>) {
            let shape = input.shape().to_vec();
            let last_dim = shape[shape.len() - 1];
            let batch_size: usize = shape[..shape.len() - 1].iter().product();

            let mut input_2d = input
                .view_mut()
                .into_shape((batch_size, last_dim))
                .unwrap();

            for mut row in input_2d.rows_mut() {
                let mean = row.mean().unwrap();
                let var = row.mapv(|v| (v - mean) * (v - mean)).mean().unwrap();
                let inv_std = (var + self.eps).rsqrt();

                for (i, val) in row.iter_mut().enumerate() {
                    *val = (*val - mean) * inv_std * self.weight[i] + self.bias[i];
                }
            }
        }
    }
}

/// Embedding layer
#[derive(Debug, Clone)]
pub struct Embedding {
    pub weight: ArrayD<f32>,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self {
            weight: ArrayD::zeros(IxDyn(&[num_embeddings, embedding_dim])),
            num_embeddings,
            embedding_dim,
        }
    }

    pub fn forward(&self, input: &ArrayD<u32>) -> Result<ArrayD<f32>, TensorError> {
        let shape = input.shape();
        let mut output_shape = shape.to_vec();
        output_shape.push(self.embedding_dim);

        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        // Flatten input for indexing
        let flat_input = input.iter().cloned().collect::<Vec<_>>();
        let flat_output = output.as_slice_mut().unwrap();

        for (i, &idx) in flat_input.iter().enumerate() {
            if idx as usize >= self.num_embeddings {
                return Err(TensorError::InvalidShape(format!(
                    "Embedding index {} out of bounds",
                    idx
                )));
            }
            let src_start = (idx as usize) * self.embedding_dim;
            let dst_start = i * self.embedding_dim;
            flat_output[dst_start..dst_start + self.embedding_dim]
                .copy_from_slice(&self.weight.as_slice().unwrap()[src_start..src_start + self.embedding_dim]);
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_weight_conversion() {
        assert_eq!(TernaryWeight::from_f32(0.5, 0.1), TernaryWeight::Pos);
        assert_eq!(TernaryWeight::from_f32(-0.5, 0.1), TernaryWeight::Neg);
        assert_eq!(TernaryWeight::from_f32(0.01, 0.1), TernaryWeight::Zero);
    }

    #[test]
    fn test_packed_tensor() {
        let weights = vec![
            TernaryWeight::Pos,
            TernaryWeight::Neg,
            TernaryWeight::Zero,
            TernaryWeight::Pos,
            TernaryWeight::Neg,
        ];
        let packed = PackedTernaryTensor::new(&weights, vec![5]);
        assert_eq!(packed.len(), 5);
        assert_eq!(packed.get(0), Some(TernaryWeight::Pos));
        assert_eq!(packed.get(1), Some(TernaryWeight::Neg));
        assert_eq!(packed.get(2), Some(TernaryWeight::Zero));
        assert_eq!(packed.get(3), Some(TernaryWeight::Pos));
        assert_eq!(packed.get(4), Some(TernaryWeight::Neg));
    }

    #[test]
    fn test_ternary_linear() {
        let weights = vec![
            TernaryWeight::Pos, TernaryWeight::Neg, // neuron 0
            TernaryWeight::Zero, TernaryWeight::Pos, // neuron 1
        ];
        let packed = PackedTernaryTensor::new(&weights, vec![2, 2]);
        let layer = TernaryLinear::new(
            packed,
            None,
            TernaryQuantParams::default(),
        ).unwrap();

        let input = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap();
        let output = layer.forward(&input).unwrap();

        // Neuron 0: 1*1 + 2*(-1) = -1
        // Neuron 1: 1*0 + 2*1 = 2
        assert!((output[[0, 0]] - (-1.0)).abs() < 1e-5);
        assert!((output[[0, 1]] - 2.0).abs() < 1e-5);
    }
}