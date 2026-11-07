use clap::{Parser, Subcommand};
use mimalloc::MiMalloc;
use parking_lot::RwLock;
use rust_vision_engine::{
    api::run_http_server,
    batcher::BatchCoordinator,
    cli::run_bulk_directory_scan,
    config::{EngineConfig, OperationalMode},
    engine::{FlatVectorIndex, SpeculativeCascadeEngine},
    error::VisionError,
};
use std::io::Cursor;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Debug)]
#[command(
    name = "rust-vision-engine",
    version,
    about = "Ultra-low-latency SIMD & ONNX hardware-accelerated image recognition engine"
)]
struct Cli {
    /// Path to primary Tier-1 ONNX model
    #[arg(long, env = "VISION_TIER1_MODEL")]
    tier1_model: Option<PathBuf>,

    /// Path to heavy Tier-2 ONNX model for speculative cascade fallback
    #[arg(long, env = "VISION_TIER2_MODEL")]
    tier2_model: Option<PathBuf>,

    /// Optional newline-separated class labels matching the model output order
    #[arg(long, env = "VISION_LABELS")]
    labels_file: Option<PathBuf>,

    /// Maximum dynamic batch size
    #[arg(long, default_value_t = 32)]
    max_batch_size: usize,

    /// Number of reusable inference tensor buffers to keep in the pool
    #[arg(long, default_value_t = 16)]
    tensor_pool_capacity: usize,

    /// ONNX CPU inference threads. Lower values reduce CPU and power use.
    #[arg(long, global = true, default_value_t = 2)]
    intra_op_threads: usize,

    /// Output dimension for the configured image embedding model
    #[arg(long, env = "VISION_EMBEDDING_DIM", default_value_t = 384)]
    embedding_dim: usize,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Bulk scan a directory of 1,000+ photos using parallel memory-mapped SIMD + batched inference
    Scan {
        /// Input directory containing photos (JPEG, PNG, WebP)
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Output NDJSON file path
        #[arg(short, long, default_value = "scan_results.ndjson")]
        output: PathBuf,

        /// Top-K predictions per image
        #[arg(short, long, default_value_t = 5)]
        top_k: usize,
    },
    /// Build a zero-copy `.rkyv` SIMD vector index from a folder of reference photos
    Index {
        /// Directory of reference photos to embed
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Output `.rkyv` index file
        #[arg(short = 'o', long, default_value = "vectors.rkyv")]
        index_file: PathBuf,
    },
    /// Export photo IDs and vector embeddings from a saved index as JSON
    ExportVectors {
        /// Existing `.rkyv` index file
        #[arg(long)]
        index_file: PathBuf,

        /// Output JSON path
        #[arg(short, long, default_value = "photo_vectors.json")]
        output: PathBuf,
    },
    /// Start the low-latency production HTTP server
    Serve {
        /// Socket address to bind
        #[arg(short, long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,

        /// Inference operation enabled by this server instance
        #[arg(long, value_enum, default_value = "speculative-cascade")]
        mode: OperationalMode,

        /// Optional pre-built `.rkyv` vector index file to load on startup
        #[arg(long)]
        index_file: Option<PathBuf>,

        /// Folder containing the indexed photos; enables result thumbnails in the test page
        #[arg(long)]
        image_dir: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let mut config = EngineConfig {
        tier1_model_path: cli.tier1_model,
        tier2_model_path: cli.tier2_model,
        labels_path: cli.labels_file,
        max_batch_size: cli.max_batch_size,
        tensor_pool_capacity: cli.tensor_pool_capacity,
        intra_op_threads: cli.intra_op_threads,
        embedding_dim: cli.embedding_dim,
        ..EngineConfig::default()
    };

    match cli.command {
        Commands::Scan {
            input_dir,
            output,
            top_k,
        } => {
            config.mode = OperationalMode::SpeculativeCascade;
            let engine = Arc::new(SpeculativeCascadeEngine::new(config)?);
            let summary = run_bulk_directory_scan(engine, &input_dir, &output, top_k)?;
            tracing::info!(
                discovered = summary.total_files_discovered,
                succeeded = summary.successful_images,
                failed = summary.failed_images,
                wall_time_ms = format!("{:.2}", summary.wall_time_ms),
                fps = format!("{:.1}", summary.throughput_images_per_sec),
                "Bulk scan completed"
            );
        }
        Commands::Index {
            input_dir,
            index_file,
        } => {
            config.mode = OperationalMode::EmbeddingSearch;
            let engine = Arc::new(SpeculativeCascadeEngine::new(config.clone())?);
            let mut v_index = FlatVectorIndex::new(config.embedding_dim);
            let start = Instant::now();

            for entry in WalkDir::new(&input_dir).into_iter() {
                let entry = entry
                    .map_err(|e| anyhow::anyhow!("failed to traverse input directory: {e}"))?;
                if !entry.file_type().is_file() {
                    continue;
                }
                let path = entry.path();
                let is_image = path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| {
                        matches!(
                            ext.to_ascii_lowercase().as_str(),
                            "jpg" | "jpeg" | "png" | "webp"
                        )
                    })
                    .unwrap_or(false);
                if !is_image {
                    continue;
                }
                let id = path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(str::to_owned)
                    .unwrap_or_else(|| path.to_string_lossy().into_owned());
                let bytes = std::fs::read(path)?;
                let views = photo_views(&bytes).map_err(|e| {
                    anyhow::anyhow!("failed to create crop views for {}: {e}", path.display())
                })?;
                for view in views {
                    let (embedding, _) = engine
                        .embed_single(&view)
                        .map_err(|e| anyhow::anyhow!("failed to embed {}: {e}", path.display()))?;
                    v_index.insert(id.clone(), embedding)?;
                }
            }

            v_index.save_to_disk(&index_file)?;
            tracing::info!(
                indexed_photos = v_index.photo_count(),
                indexed_views = v_index.len(),
                elapsed_ms = format!("{:.2}", start.elapsed().as_secs_f64() * 1000.0),
                output = ?index_file,
                "Saved rkyv SIMD vector index"
            );
        }
        Commands::ExportVectors { index_file, output } => {
            let index = FlatVectorIndex::load_from_disk(&index_file)?;
            let view_names = ["full", "center", "top", "bottom"];
            let mut view_counts = std::collections::HashMap::<String, usize>::new();
            let rows: Vec<serde_json::Value> = index
                .iter_vectors()
                .map(|(id, vector)| {
                    let view_number = view_counts.entry(id.to_owned()).or_default();
                    let view_name = view_names.get(*view_number).copied().unwrap_or("view");
                    let vector_id = format!("{id}-{view_name}-{:02}", *view_number);
                    *view_number += 1;
                    serde_json::json!({
                        "product_id": id,
                        "product_name": format!("Product {id}"),
                        "vector_id": vector_id,
                        "vector_indexing_data": vector,
                    })
                })
                .collect();
            let file = std::fs::File::create(&output)?;
            serde_json::to_writer(file, &rows)?;
            tracing::info!(
                product_count = index.photo_count(),
                vector_count = rows.len(),
                dimension = index.dim(),
                output = ?output,
                "Exported indexed vectors"
            );
        }
        Commands::Serve {
            bind,
            index_file,
            image_dir,
            mode,
        } => {
            config.mode = mode;
            let engine = Arc::new(SpeculativeCascadeEngine::new(config.clone())?);
            let batch_handle = BatchCoordinator::spawn(Arc::clone(&engine), 1024);

            let loaded_index = match &index_file {
                Some(path) if path.exists() => FlatVectorIndex::load_from_disk(path)?,
                _ => FlatVectorIndex::new(config.embedding_dim),
            };
            if loaded_index.dim() != config.embedding_dim {
                return Err(anyhow::anyhow!(
                    "loaded vector index dimension {} does not match configured embedding dimension {}",
                    loaded_index.dim(),
                    config.embedding_dim
                ));
            }
            let shared_index = Arc::new(RwLock::new(loaded_index));
            run_http_server(bind, engine, batch_handle, shared_index, image_dir)
                .await
                .map_err(|e: VisionError| anyhow::anyhow!(e))?;
        }
    }

    Ok(())
}

/// Returns the original photo plus three overlapping crops for crop-tolerant matching.
fn photo_views(bytes: &[u8]) -> anyhow::Result<Vec<Vec<u8>>> {
    let image = image::load_from_memory(bytes)?;
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
        crop.write_to(&mut encoded, image::ImageFormat::Png)?;
        views.push(encoded.into_inner());
    }
    Ok(views)
}
