//! OpenAI-compatible API server

use crate::engine::{EngineFactory, InferenceConfig, InferenceEngine, InferenceInput};
use crate::model::{ModelManager, ModelRecommendation};
use anyhow::{Context, Result};
use axum::{
    extract::{Extension, Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{Json, Sse, IntoResponse},
    routing::{get, post},
    Router,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, Mutex, RwLock};
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, info, warn};

/// API server state
#[derive(Clone)]
pub struct ApiState {
    pub model_manager: Arc<ModelManager>,
    pub engine: Arc<Mutex<Box<dyn InferenceEngine>>>,
    pub config: ApiConfig,
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Server bind address
    pub bind_address: String,
    /// Server port
    pub port: u16,
    /// API key (optional)
    pub api_key: Option<String>,
    /// Enable CORS
    pub cors: bool,
    /// Request timeout (seconds)
    pub timeout_seconds: u64,
    /// Max concurrent requests
    pub max_concurrent_requests: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1".to_string(),
            port: 8080,
            api_key: None,
            cors: true,
            timeout_seconds: 300,
            max_concurrent_requests: 100,
        }
    }
}

/// Chat completion request (OpenAI compatible)
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model ID
    pub model: String,
    /// Messages
    pub messages: Vec<ChatMessage>,
    /// Temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-p
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Top-k
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Max tokens
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Stream response
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// Presence penalty
    #[serde(default)]
    pub presence_penalty: f32,
    /// Frequency penalty
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Random seed
    #[serde(default)]
    pub seed: Option<u64>,
    /// User ID
    #[serde(default)]
    pub user: Option<String>,
}

fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }

/// Chat message
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub name: Option<String>,
}

/// Chat completion response
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// Chat choice
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Usage statistics
#[derive(Debug, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming chat completion chunk
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// Streaming chunk choice
#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

/// Delta for streaming
#[derive(Debug, Serialize, Default)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Models list response
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model info
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub engine: String,
    pub model_loaded: Option<String>,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// Create the API router
pub fn create_router(state: ApiState) -> Router {
    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/models/:model_id", get(get_model))
        .route("/health", get(health_check))
        .route("/v1/engines", get(list_engines))
        .route("/v1/recommendations", get(get_recommendations))
        .with_state(state);

    // Add CORS if enabled
    let state_clone = router.state().unwrap().downcast_ref::<ApiState>().unwrap();
    if state_clone.config.cors {
        router = router.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    }

    router
}

/// Chat completions endpoint
async fn chat_completions(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Check API key
    if let Some(api_key) = &state.config.api_key {
        if let Some(auth) = headers.get("authorization") {
            if let Ok(auth_str) = auth.to_str() {
                if !auth_str.starts_with("Bearer ") || &auth_str[7..] != api_key {
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(ErrorResponse {
                            error: ErrorDetail {
                                message: "Invalid API key".to_string(),
                                r#type: "auth_error".to_string(),
                                param: None,
                                code: Some("invalid_api_key".to_string()),
                            },
                        }),
                    ).into_response();
                }
            }
        } else {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "Missing API key".to_string(),
                        r#type: "auth_error".to_string(),
                        param: None,
                        code: Some("missing_api_key".to_string()),
                    },
                }),
            ).into_response();
        }
    }

    // Validate request
    if request.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "Messages cannot be empty".to_string(),
                    r#type: "invalid_request_error".to_string(),
                    param: Some("messages".to_string()),
                    code: None,
                },
            }),
        ).into_response();
    }

    // Load model if not loaded
    let model_loaded = {
        let engine = state.engine.lock().await;
        engine.model().map(|m| m.metadata.id.clone())
    };

    if model_loaded.as_ref() != Some(&request.model) {
        // Try to load the requested model
        let model_manager = state.model_manager.clone();
        let model_id = request.model.clone();
        let config = InferenceConfig {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k.unwrap_or(50),
            seed: request.seed,
            ..Default::default()
        };

        if let Err(e) = model_manager.load_model(&model_id, Some(config)).await {
            warn!("Failed to load model {}: {}", model_id, e);
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Model '{}' not found: {}", model_id, e),
                        r#type: "not_found_error".to_string(),
                        param: Some("model".to_string()),
                        code: Some("model_not_found".to_string()),
                    },
                }),
            ).into_response();
        }

        // Re-initialize engine with new model
        let model = model_manager.load_model(&model_id, None).await.unwrap();
        let mut engine = state.engine.lock().await;
        if let Err(e) = engine.initialize(model).await {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Failed to initialize engine: {}", e),
                        r#type: "internal_error".to_string(),
                        param: None,
                        code: Some("engine_init_failed".to_string()),
                    },
                }),
            ).into_response();
        }
    }

    // Convert messages to tokens (simplified)
    let prompt = messages_to_prompt(&request.messages);
    let tokens = tokenize_simple(&prompt); // Would use actual tokenizer

    let input = InferenceInput {
        tokens,
        config: InferenceConfig {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k.unwrap_or(50),
            max_tokens: request.max_tokens.unwrap_or(1024),
            seed: request.seed,
            ..Default::default()
        },
        kv_cache: None,
    };

    if request.stream {
        stream_response(state, request.model, input).await.into_response()
    } else {
        sync_response(state, request.model, input).await.into_response()
    }
}

/// Synchronous (non-streaming) response
async fn sync_response(
    state: ApiState,
    model: String,
    input: InferenceInput,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut engine = state.engine.lock().await;
    let output = engine.infer(&input).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Inference failed: {}", e),
                    r#type: "internal_error".to_string(),
                    param: None,
                    code: Some("inference_failed".to_string()),
                },
            }),
        )
    })?;

    // Decode tokens to text (simplified)
    let generated_text = decode_simple(&output.logits); // Would use actual tokenizer

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: generated_text,
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: input.tokens.len(),
            completion_tokens: 100, // Estimated
            total_tokens: input.tokens.len() + 100,
        },
    };

    Ok(Json(response))
}

/// Streaming response
async fn stream_response(
    state: ApiState,
    model: String,
    input: InferenceInput,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, axum::Error>>> {
    let (tx, rx) = broadcast::channel(100);
    let model_clone = model.clone();
    let engine = state.engine.clone();

    // Spawn inference task
    tokio::spawn(async move {
        let mut engine = engine.lock().await;
        let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();

        // First chunk with role
        let first_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_clone.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let _ = tx.send(Ok(axum::response::sse::Event::default().json_data(first_chunk)));

        // Stream tokens
        let mut current_input = input;
        loop {
            let output = match engine.infer(&current_input).await {
                Ok(o) => o,
                Err(e) => {
                    let _ = tx.send(Err(axum::Error::new(e)));
                    break;
                }
            };

            // Sample token
            let mut rng = rand::rngs::StdRng::from_entropy();
            let token = crate::engine::TernaryEngine::sample_token(&engine, &output.logits, &current_input.config, &mut rng);

            // Decode token
            let text = decode_token(token);

            let chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_clone.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: Some(text),
                    },
                    finish_reason: if token == 0 || token == 2 { Some("stop".to_string()) } else { None },
                }],
            };

            let is_final = token == 0 || token == 2;
            let _ = tx.send(Ok(axum::response::sse::Event::default().json_data(chunk)));

            if is_final {
                break;
            }

            current_input.tokens = vec![token];
            current_input.kv_cache = Some(output.kv_cache);
        }

        // Final [DONE] message
        let _ = tx.send(Ok(axum::response::sse::Event::default().data("[DONE]")));
    });

    // Convert broadcast receiver to stream
    let stream = stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Ok(event) => Some((event, rx)),
            Err(_) => None,
        }
    });

    Sse::new(stream)
}

/// List models endpoint
async fn list_models(State(state): State<ApiState>) -> Json<ModelsResponse> {
    let model_manager = state.model_manager.clone();
    let loaded = model_manager.list_loaded_models().await;

    let models: Vec<ModelInfo> = loaded
        .into_iter()
        .map(|id| ModelInfo {
            id: id.clone(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "tern".to_string(),
        })
        .collect();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

/// Get model endpoint
async fn get_model(
    State(state): State<ApiState>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelInfo>, (StatusCode, Json<ErrorResponse>)> {
    let model_manager = state.model_manager.clone();

    if let Ok(model) = model_manager.load_model(&model_id, None).await {
        Ok(Json(ModelInfo {
            id: model.metadata.id.clone(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "tern".to_string(),
        }))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Model '{}' not found", model_id),
                    r#type: "not_found_error".to_string(),
                    param: Some("model".to_string()),
                    code: Some("model_not_found".to_string()),
                },
            }),
        ))
    }
}

/// Health check endpoint
async fn health_check(State(state): State<ApiState>) -> Json<HealthResponse> {
    let engine = state.engine.lock().await;
    let model_loaded = engine.model().map(|m| m.metadata.id.clone());

    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        engine: engine.name().to_string(),
        model_loaded,
    })
}

/// List engines endpoint
async fn list_engines() -> Json<Vec<String>> {
    Json(EngineFactory::available_engines().into_iter().map(|s| s.to_string()).collect())
}

/// Get recommendations endpoint
async fn get_recommendations(State(state): State<ApiState>) -> Json<Vec<ModelRecommendation>> {
    let recommendations = state.model_manager.recommend_models();
    Json(recommendations)
}

/// Simple message to prompt conversion
fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => prompt.push_str(&format!("<|system|>\n{}\n", msg.content)),
            "user" => prompt.push_str(&format!("<|user|>\n{}\n", msg.content)),
            "assistant" => prompt.push_str(&format!("<|assistant|>\n{}\n", msg.content)),
            _ => prompt.push_str(&format!("<|{}|\n{}\n", msg.role, msg.content)),
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Simple tokenization (placeholder)
fn tokenize_simple(text: &str) -> Vec<u32> {
    // Very simple tokenization - in reality would use actual tokenizer
    text.split_whitespace()
        .enumerate()
        .map(|(i, _)| i as u32 % 32000)
        .collect()
}

/// Simple decoding (placeholder)
fn decode_simple(logits: &ndarray::ArrayD<f32>) -> String {
    // Would use actual tokenizer decode
    "Generated response".to_string()
}

/// Decode single token (placeholder)
fn decode_token(token: u32) -> String {
    // Would use actual tokenizer decode
    format!("token_{} ", token)
}

/// Run the API server
pub async fn run_server(config: ApiConfig, model_manager: Arc<ModelManager>, engine_name: &str) -> Result<()> {
    let engine_config = InferenceConfig::default();
    let engine = EngineFactory::create(engine_name, engine_config)
        .context("Failed to create engine")?;

    let state = ApiState {
        model_manager,
        engine: Arc::new(Mutex::new(engine)),
        config: config.clone(),
    };

    let app = create_router(state);

    let addr = format!("{}:{}", config.bind_address, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await
        .context("Failed to bind to address")?;

    info!("API server listening on {}", addr);

    axum::serve(listener, app).await
        .context("Server error")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_messages_to_prompt() {
        let messages = vec![
            ChatMessage { role: "system".to_string(), content: "You are helpful".to_string(), name: None },
            ChatMessage { role: "user".to_string(), content: "Hello".to_string(), name: None },
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("<|assistant|>"));
    }

    #[test]
    fn test_chat_request_defaults() {
        let request: ChatCompletionRequest = serde_json::from_str(r#"{"model": "test", "messages": [{"role": "user", "content": "hi"}]}"#).unwrap();
        assert_eq!(request.temperature, 0.7);
        assert_eq!(request.top_p, 0.9);
        assert!(!request.stream);
    }
}