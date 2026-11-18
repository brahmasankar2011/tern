//! Integration tests for Ternary LLM Inference

use tern_core::{
    engine::{EngineFactory, InferenceConfig},
    hardware::HardwareDetector,
    model::{ModelManager, InferenceConfig as ModelInferenceConfig},
    tensor::{PackedTernaryTensor, TernaryLinear, TernaryQuantParams, TernaryWeight, activation, layer_norm},
};

#[tokio::test]
async fn test_hardware_detection() {
    let mut detector = HardwareDetector::new();
    let hw = detector.detect();

    assert!(!hw.cpu.brand.is_empty());
    assert!(hw.cpu.physical_cores > 0);
    assert!(hw.memory.total_bytes > 0);
    assert!(!hw.os.name.is_empty());
    assert!(hw.recommended_threads > 0);
}

#[tokio::test]
async fn test_model_recommendations() {
    let mut detector = HardwareDetector::new();
    let hw = detector.detect();
    let temp_dir = tempfile::TempDir::new().unwrap();
    let manager = ModelManager::new(temp_dir.path().to_path_buf(), hw).unwrap();

    let recommendations = manager.recommend_models();
    assert!(!recommendations.is_empty());

    // Should have at least some models
    for rec in recommendations {
        assert!(!rec.model_id.is_empty());
        assert!(rec.size_b > 0);
        assert!(rec.estimated_ram_gb > 0.0);
    }
}

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
        TernaryWeight::Pos, TernaryWeight::Neg,
        TernaryWeight::Zero, TernaryWeight::Pos,
    ];
    let packed = PackedTernaryTensor::new(&weights, vec![2, 2]);
    let layer = TernaryLinear::new(
        packed,
        None,
        TernaryQuantParams::default(),
    ).unwrap();

    let input = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 2]),
        vec![1.0, 2.0],
    ).unwrap();
    let output = layer.forward(&input).unwrap();

    assert!((output[[0, 0]] - (-1.0)).abs() < 1e-5);
    assert!((output[[0, 1]] - 2.0).abs() < 1e-5);
}

#[test]
fn test_activation_functions() {
    use ndarray::ArrayD;

    // Test ReLU
    let mut x = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap();
    activation::relu(&mut x);
    assert_eq!(x.as_slice().unwrap(), &[0.0, 0.0, 1.0]);

    // Test GELU
    let mut x = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap();
    activation::gelu(&mut x);
    let vals: Vec<f32> = x.as_slice().unwrap().iter().map(|v| *v).collect();
    assert!(vals[0] < 0.0);
    assert!((vals[1] - 0.0).abs() < 1e-5);
    assert!(vals[2] > 0.0);

    // Test SiLU
    let mut x = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap();
    activation::silu(&mut x);
    let vals: Vec<f32> = x.as_slice().unwrap().iter().map(|v| *v).collect();
    assert!(vals[0] < 0.0);
    assert!((vals[1] - 0.0).abs() < 1e-5);
    assert!(vals[2] > 0.0);
}

#[test]
fn test_layer_norm() {
    use ndarray::ArrayD;

    let ln = layer_norm::LayerNorm::new(4, 1e-6);
    let mut x = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 4]),
        vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    ).unwrap();

    ln.forward(&mut x);

    // Check mean ~0 and std ~1 for each row
    for row in x.rows() {
        let mean = row.mean().unwrap();
        let var = row.mapv(|v| (v - mean) * (v - mean)).mean().unwrap();
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_softmax() {
    use ndarray::ArrayD;

    let mut x = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 3]),
        vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
    ).unwrap();

    activation::softmax(&mut x);

    for row in x.rows() {
        let sum: f32 = row.sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(row.iter().all(|&v| v >= 0.0));
    }
}

#[test]
fn test_engine_factory() {
    let config = InferenceConfig::default();
    let engine = EngineFactory::create("ternary", config).unwrap();
    assert_eq!(engine.name(), "ternary-engine");
    assert!(engine.supported_architectures().contains(&"bitnet"));
}

#[test]
fn test_kv_cache() {
    let cache = tern_core::engine::KVCache::new(32, 1, 32, 128, 4096);
    assert_eq!(cache.keys.len(), 32);
    assert_eq!(cache.values.len(), 32);
}

#[test]
fn test_causal_mask() {
    let engine = tern_core::engine::TernaryEngine::new(InferenceConfig::default());
    let mask = engine.create_causal_mask(1, 4, 4).unwrap();
    assert_eq!(mask.shape(), &[1, 1, 4, 4]);

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

#[test]
fn test_utils() {
    assert_eq!(tern_core::utils::format_bytes(500), "500 B");
    assert_eq!(tern_core::utils::format_bytes(1024), "1.00 KB");
    assert_eq!(tern_core::utils::format_bytes(1024 * 1024), "1.00 MB");

    assert!(tern_core::utils::validate_model_id("user/model").is_ok());
    assert!(tern_core::utils::validate_model_id("model-name_v2.1").is_ok());
    assert!(tern_core::utils::validate_model_id("").is_err());
    assert!(tern_core::utils::validate_model_id("model@name").is_err());

    assert_eq!(tern_core::utils::parse_csv_list("a,b,c"), vec!["a", "b", "c"]);
    assert_eq!(tern_core::utils::truncate("hello", 10), "hello");
    assert_eq!(tern_core::utils::truncate("hello world", 8), "hello...");
}