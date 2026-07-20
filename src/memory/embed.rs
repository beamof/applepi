//! 本地 embedding：基于 fastembed (ONNX Runtime) 的进程级单例。
//!
//! 设计要点：
//! - **单例 + 启动预热**：模型加载只发生一次（首次 `global()` 调用，约 200ms~1s），
//!   之后所有 Agent / LongTermMemory 共享同一个 `TextEmbedding`，避免重复加载占内存。
//! - **CPU offload**：`TextEmbedding::embed` 是阻塞 CPU 调用，用 `spawn_blocking` 包裹，
//!   避免阻塞 tokio runtime。
//! - **互斥**：fastembed 的 `embed` 需要 `&mut self`，用 `std::sync::Mutex` 串行化。
//!   单次推理 <30ms，串行不会成为瓶颈（recall 一次只 embed 一条 query）。

use anyhow::{anyhow, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// 本地 embedding 推理器（进程单例）。
///
/// 持有一个加载好的 `TextEmbedding`，所有调用共享。
pub struct LocalEmbedder {
    /// fastembed 内部 embed 需要 &mut self，用同步 Mutex 串行化。
    /// 持锁的临界区在 `spawn_blocking` 内执行，不阻塞 tokio runtime 线程。
    model: Mutex<TextEmbedding>,
    /// 向量维度（加载时确定，供调用方校验/调试）。
    pub dim: usize,
}

static INSTANCE: OnceLock<Result<LocalEmbedder>> = OnceLock::new();

/// 选中的本地模型（中文场景默认 BGE-small-zh-v1.5，512 维，~100MB）。
fn pick_model(name: &str) -> Result<EmbeddingModel> {
    let n = name.trim().to_ascii_lowercase();
    let m = match n.as_str() {
        "" | "bge-small-zh-v1.5" | "bge-small-zh" => EmbeddingModel::BGESmallZHV15,
        "bge-large-zh-v1.5" | "bge-large-zh" => EmbeddingModel::BGELargeZHV15,
        "bge-small-en-v1.5" | "bge-small-en" => EmbeddingModel::BGESmallENV15,
        "bge-base-en-v1.5" | "bge-base-en" => EmbeddingModel::BGEBaseENV15,
        "multilingual-e5-small" => EmbeddingModel::MultilingualE5Small,
        "multilingual-e5-base" => EmbeddingModel::MultilingualE5Base,
        "multilingual-e5-large" => EmbeddingModel::MultilingualE5Large,
        "bge-m3" | "bgem3" => EmbeddingModel::BGEM3,
        other => {
            return Err(anyhow!(
                "未知 embeddings.model '{other}'；可用：bge-small-zh-v1.5 | bge-large-zh-v1.5 \
                 | bge-small-en-v1.5 | bge-base-en-v1.5 | multilingual-e5-small \
                 | multilingual-e5-base | multilingual-e5-large | bge-m3"
            ));
        }
    };
    Ok(m)
}

impl LocalEmbedder {
    /// 获取进程级单例。首次调用会触发模型加载（同步、可能下载 ~100MB 模型文件）。
    /// 后续调用直接返回缓存。
    ///
    /// 返回 `&'static` 的 `Result`：加载失败也会被缓存，避免每次都重试（用户需重启进程）。
    pub fn global(model_name: &str, cache_dir: Option<&str>) -> Result<&'static LocalEmbedder> {
        INSTANCE
            .get_or_init(|| Self::build(model_name, cache_dir))
            .as_ref()
            .map_err(|e| anyhow!("本地 embedding 模型初始化失败: {e}"))
    }

    fn build(model_name: &str, cache_dir: Option<&str>) -> Result<LocalEmbedder> {
        let model_enum = pick_model(model_name)?;
        let dim = TextEmbedding::get_model_info(&model_enum)
            .map_err(|e| anyhow!("找不到模型维度元数据: {e}"))?
            .dim;

        let mut opts = TextInitOptions::new(model_enum).with_show_download_progress(true);
        if let Some(dir) = cache_dir.filter(|s| !s.trim().is_empty()) {
            opts = opts.with_cache_dir(PathBuf::from(dir));
        }
        let model = TextEmbedding::try_new(opts)?;
        tracing::info!(
            "本地 embedding 模型加载完成：{model_name}（dim={dim}）。后续推理离线进行。"
        );
        Ok(LocalEmbedder {
            model: Mutex::new(model),
            dim,
        })
    }

    /// 单条文本 embedding。
    ///
    /// 仅可通过 `global()` 返回的 `&'static Self` 调用，这样 `spawn_blocking` 闭包能持有引用。
    pub async fn embed(self: &'static Self, text: String) -> Result<Vec<f32>> {
        tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
            let mut m = self.model.lock().expect("LocalEmbedder mutex poisoned");
            let mut out = m.embed(vec![text], None)?;
            out.pop().ok_or_else(|| anyhow!("embedding 返回空"))
        })
        .await
        .map_err(|e| anyhow!("embed 任务失败: {e}"))?
    }

    /// 批量 embedding（迁移用）。
    pub async fn embed_batch(self: &'static Self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
            let mut m = self.model.lock().expect("LocalEmbedder mutex poisoned");
            let out = m.embed(texts, None)?;
            Ok(out)
        })
        .await
        .map_err(|e| anyhow!("embed_batch 任务失败: {e}"))?
    }
}
