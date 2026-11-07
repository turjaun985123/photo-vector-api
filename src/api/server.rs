use crate::batcher::BatchHandle;
use crate::config::LatencyBreakdown;
use crate::engine::{FlatVectorIndex, SearchMatch, SpeculativeCascadeEngine};
use crate::error::VisionError;
use crate::pipeline::postprocess::{ClassificationResult, DetectionResult};
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use metrics::{counter, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Cursor;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;
use walkdir::WalkDir;

pub const API_TOKEN: &str = "npc-image-index-dev-token";

/// Shared application state across Axum worker threads.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<SpeculativeCascadeEngine>,
    pub batch_handle: BatchHandle,
    pub vector_index: Arc<RwLock<FlatVectorIndex>>,
    pub prometheus_handle: Option<PrometheusHandle>,
    pub photo_catalog: Arc<HashMap<String, PathBuf>>,
    pub api_token: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct ProductVector {
    vector_id: String,
    vector_indexing_data: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct ProductUploadResponse {
    request_id: String,
    vectors: Vec<ProductVector>,
}

struct UploadedPhoto {
    bytes: Vec<u8>,
}

#[derive(Debug, Deserialize)]
pub struct TopKQuery {
    pub top_k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ClassifyApiResponse {
    pub result: ClassificationResult,
    pub latency: LatencyBreakdown,
}

#[derive(Debug, Serialize)]
pub struct DetectApiResponse {
    pub result: DetectionResult,
    pub latency: LatencyBreakdown,
}

#[derive(Debug, Serialize)]
pub struct SearchApiResponse {
    pub matches: Vec<SearchMatch>,
    pub index_size: usize,
    pub search_us: u64,
    pub embed_latency: LatencyBreakdown,
}

impl IntoResponse for VisionError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::InvalidHeader(_)
            | Self::ImageTooLarge { .. }
            | Self::ZeroDimension { .. }
            | Self::JpegDecode(_)
            | Self::FallbackDecode(_)
            | Self::Config(_) => StatusCode::BAD_REQUEST,
            Self::PoolExhausted { .. } | Self::BatcherClosed(_) => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = serde_json::json!({
            "error": self.to_string(),
        });
        (status, Json(body)).into_response()
    }
}

/// Constructs the Axum Router with the visual-search test page and API endpoints.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(test_page_handler))
        .route("/test", get(test_page_handler))
        .route("/products", get(product_upload_page_handler))
        .route("/healthz", get(healthz_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/classify", post(classify_handler))
        .route("/v1/detect", post(detect_handler))
        .route("/v1/products", post(product_upload_handler))
        .route("/v1/search", post(search_handler))
        .route("/v1/photos/:id", get(photo_handler))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024)) // 32 MB max upload
        .with_state(state)
}

/// Starts the low-latency Axum HTTP daemon.
pub async fn run_http_server(
    addr: SocketAddr,
    engine: Arc<SpeculativeCascadeEngine>,
    batch_handle: BatchHandle,
    vector_index: Arc<RwLock<FlatVectorIndex>>,
    image_dir: Option<PathBuf>,
) -> Result<(), VisionError> {
    let photo_catalog = Arc::new(load_photo_catalog(image_dir.as_deref())?);
    let prometheus_handle = PrometheusBuilder::new().install_recorder().ok();
    let state = AppState {
        engine,
        batch_handle,
        vector_index,
        prometheus_handle,
        photo_catalog,
        api_token: API_TOKEN,
    };

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|source| VisionError::Io { path: None, source })?;

    tracing::info!("rust-vision-engine HTTP daemon listening on http://{addr}");
    axum::serve(listener, app)
        .await
        .map_err(|source| VisionError::Io { path: None, source })
}

fn load_photo_catalog(image_dir: Option<&Path>) -> Result<HashMap<String, PathBuf>, VisionError> {
    let Some(image_dir) = image_dir else {
        return Ok(HashMap::new());
    };
    if !image_dir.is_dir() {
        return Err(VisionError::Config(format!(
            "photo directory does not exist or is not a directory: {}",
            image_dir.display()
        )));
    }

    let mut photos = HashMap::new();
    for entry in WalkDir::new(image_dir) {
        let entry = entry.map_err(|error| {
            VisionError::Config(format!("failed to scan photo directory: {error}"))
        })?;
        if !entry.file_type().is_file() || !is_supported_image(entry.path()) {
            continue;
        }
        let Some(id) = entry.path().file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if photos
            .insert(id.to_owned(), entry.path().to_path_buf())
            .is_some()
        {
            return Err(VisionError::Config(format!(
                "photo directory contains duplicate filename ID '{id}'"
            )));
        }
    }
    Ok(photos)
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp"
            )
        })
        .unwrap_or(false)
}

async fn test_page_handler() -> Html<&'static str> {
    Html(include_str!("../../static/test.html"))
}

async fn product_upload_page_handler() -> Html<&'static str> {
    Html(include_str!("../../static/product_upload.html"))
}

async fn photo_handler(State(state): State<AppState>, AxumPath(id): AxumPath<String>) -> Response {
    let Some(path) = state.photo_catalog.get(&id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let bytes = match tokio::fs::read(&path).await {
        Ok(bytes) => bytes,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let content_type = match path.extension().and_then(|ext| ext.to_str()).unwrap_or("") {
        ext if ext.eq_ignore_ascii_case("png") => "image/png",
        ext if ext.eq_ignore_ascii_case("webp") => "image/webp",
        _ => "image/jpeg",
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(bytes))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

async fn healthz_handler(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "pool_available": state.engine.tier1_pool().available(),
        "indexed_vectors": state.vector_index.read().len(),
        "photo_previews_available": state.photo_catalog.len(),
    }))
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    match &state.prometheus_handle {
        Some(h) => (StatusCode::OK, h.render()),
        None => (
            StatusCode::OK,
            "# Prometheus recorder not initialized\n".to_string(),
        ),
    }
}

async fn classify_handler(
    State(state): State<AppState>,
    Query(query): Query<TopKQuery>,
    body: Bytes,
) -> Result<Json<ClassifyApiResponse>, VisionError> {
    if state.engine.config().mode != crate::config::OperationalMode::SpeculativeCascade {
        return Err(VisionError::Config(
            "classification requires speculative-cascade mode".into(),
        ));
    }
    let top_k = query.top_k.unwrap_or(5).clamp(1, 100);
    let (result, latency) = state.batch_handle.classify(body, top_k).await?;

    counter!("vision_classify_requests_total").increment(1);
    if result.resolved_by_tier2 {
        counter!("vision_cascade_tier2_escalations_total").increment(1);
    }
    histogram!("vision_classify_total_us").record(latency.total_us as f64);

    Ok(Json(ClassifyApiResponse { result, latency }))
}

async fn detect_handler(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<DetectApiResponse>, VisionError> {
    if state.engine.config().mode != crate::config::OperationalMode::ObjectDetection {
        return Err(VisionError::Config(
            "detection requires object-detection mode".into(),
        ));
    }
    let engine = Arc::clone(&state.engine);
    let (result, latency) = tokio::task::spawn_blocking(move || engine.detect_single(&body, 0))
        .await
        .map_err(|e| VisionError::Inference(e.to_string()))??;

    counter!("vision_detect_requests_total").increment(1);
    histogram!("vision_detect_total_us").record(latency.total_us as f64);

    Ok(Json(DetectApiResponse { result, latency }))
}

async fn product_upload_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    if !is_authorized(&headers, &state.api_token) {
        return api_error(StatusCode::UNAUTHORIZED, "missing or invalid bearer token");
    }
    if state.engine.config().mode != crate::config::OperationalMode::EmbeddingSearch {
        return api_error(
            StatusCode::BAD_REQUEST,
            "product indexing requires embedding-search mode",
        );
    }

    let mut photos = Vec::new();
    let mut total_bytes = 0_usize;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(error) => return api_error(StatusCode::BAD_REQUEST, &error.to_string()),
        };
        let name = field.name().unwrap_or_default().to_owned();
        if name == "images" {
            if photos.len() >= 5 {
                return api_error(StatusCode::BAD_REQUEST, "upload 4 or 5 images per product");
            }
            let supported_type = field
                .content_type()
                .map(|content_type| {
                    matches!(
                        content_type,
                        "image/jpeg" | "image/png" | "image/webp" | "application/octet-stream"
                    )
                })
                .unwrap_or(false);
            let supported_extension = field
                .file_name()
                .and_then(|name| Path::new(name).extension())
                .and_then(|extension| extension.to_str())
                .map(|extension| {
                    matches!(
                        extension.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "webp"
                    )
                })
                .unwrap_or(false);
            if !supported_type || !supported_extension {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    "images must be JPG, PNG, or WebP files",
                );
            }
            let bytes = match field.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => return api_error(StatusCode::BAD_REQUEST, &error.to_string()),
            };
            total_bytes = total_bytes.saturating_add(bytes.len());
            if bytes.is_empty() || bytes.len() > 10 * 1024 * 1024 || total_bytes > 30 * 1024 * 1024
            {
                return api_error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "each image must be under 10 MB and total images under 30 MB",
                );
            }
            photos.push(UploadedPhoto {
                bytes: bytes.to_vec(),
            });
        } else {
            return api_error(
                StatusCode::BAD_REQUEST,
                "this endpoint accepts image fields only",
            );
        }
    }

    if photos.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "upload at least one image");
    }
    match index_uploaded_photos(&state, photos).await {
        Ok(vectors) => Json(vectors).into_response(),
        Err((status, message)) => api_error(status, &message),
    }
}

async fn index_uploaded_photos(
    state: &AppState,
    photos: Vec<UploadedPhoto>,
) -> Result<ProductUploadResponse, (StatusCode, String)> {
    let engine = Arc::clone(&state.engine);
    let request_id = Uuid::new_v4().to_string();
    let request_id_for_vectors = request_id.clone();
    let vectors = tokio::task::spawn_blocking(move || {
        let mut vectors = Vec::new();
        let view_names = ["full", "center", "top", "bottom"];
        for (photo_index, photo) in photos.iter().enumerate() {
            let views = photo_views(&photo.bytes).map_err(|error| error.to_string())?;
            for (view_index, view) in views.iter().enumerate() {
                let (embedding, _) = engine
                    .embed_single(view)
                    .map_err(|error| error.to_string())?;
                let view_name = view_names.get(view_index).copied().unwrap_or("view");
                vectors.push(ProductVector {
                    vector_id: format!(
                        "{request_id_for_vectors}-image-{:04}-{view_name}",
                        photo_index + 1
                    ),
                    vector_indexing_data: embedding,
                });
            }
        }
        Ok::<_, String>(vectors)
    })
    .await
    .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?
    .map_err(|error| (StatusCode::BAD_REQUEST, error))?;
    Ok(ProductUploadResponse {
        request_id,
        vectors,
    })
}

async fn search_handler(
    State(state): State<AppState>,
    Query(query): Query<TopKQuery>,
    body: Bytes,
) -> Result<Json<SearchApiResponse>, VisionError> {
    if state.engine.config().mode != crate::config::OperationalMode::EmbeddingSearch {
        return Err(VisionError::Config(
            "search requires embedding-search mode".into(),
        ));
    }
    let top_k = query.top_k.unwrap_or(5).clamp(1, 100);
    let engine = Arc::clone(&state.engine);
    let (embedding, embed_latency) =
        tokio::task::spawn_blocking(move || engine.embed_single(&body))
            .await
            .map_err(|e| VisionError::Inference(e.to_string()))??;

    let t_search = std::time::Instant::now();
    let (matches, index_size) = {
        let idx = state.vector_index.read();
        let m = idx.search(&embedding, top_k)?;
        (m, idx.photo_count())
    };
    let search_us = t_search.elapsed().as_micros() as u64;

    Ok(Json(SearchApiResponse {
        matches,
        index_size,
        search_us,
        embed_latency,
    }))
}

fn is_authorized(headers: &HeaderMap, expected_token: &str) -> bool {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|provided| provided == expected_token)
}

fn api_error(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

fn photo_views(bytes: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let image = image::load_from_memory(bytes).map_err(|error| error.to_string())?;
    let (width, height) = image::GenericImageView::dimensions(&image);
    if width < 2 || height < 2 {
        return Ok(vec![bytes.to_vec()]);
    }
    let mut views = vec![bytes.to_vec()];
    let scale = 0.68_f32;
    let crop_width = ((width as f32 * scale).round() as u32).clamp(1, width);
    let crop_height = ((height as f32 * scale).round() as u32).clamp(1, height);
    let x = (width - crop_width) / 2;
    let max_y = height - crop_height;
    for y in [max_y / 2, 0, max_y] {
        let crop = image.crop_imm(x, y, crop_width, crop_height);
        let mut encoded = Cursor::new(Vec::new());
        crop.write_to(&mut encoded, image::ImageFormat::Png)
            .map_err(|error| error.to_string())?;
        views.push(encoded.into_inner());
    }
    Ok(views)
}
