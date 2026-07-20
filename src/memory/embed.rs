//! 本地 embedding：基于 candle（纯 Rust 推理）的进程级单例。
//!
//! 设计要点：
//! - **纯 Rust 推理**：用 `candle-transformers` 的 `BertModel` 跑 BGE / E5 模型，
//!   完全不依赖 ONNX Runtime / ort-sys，规避了 Linux 下 ort 静态库 C++ 符号
//!   (`__cxa_call_terminate`) 链接问题。BGE/E5 架构上就是 BERT + mean pooling
//!   + L2 normalize。
//! - **单例 + 启动预热**：模型加载只发生一次（首次 `global()` 调用，
//!   ~1-2s），之后所有 Agent / LongTermMemory 共享同一个模型实例。
//! - **CPU offload**：推理是阻塞 CPU 调用，用 `spawn_blocking` 包裹，避免
//!   阻塞 tokio runtime。
//! - **互斥**：candle 的 `forward` 需要 `&self` 但内部可变（算子可能写缓
//!   存），用 `std::sync::Mutex` 串行化。单次推理 ~50-80ms，串行不会成为
//!   瓶颈（recall 一次只 embed 一条 query）。
//!
//! 模型权重从 HuggingFace Hub 下载（首次），落盘到 `HF_HOME` 或配置的
//! `cache_dir`。国内网络可设 `HF_ENDPOINT=https://hf-mirror.com` 镜像。

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

/// 支持的模型短名 → (HuggingFace repo id, 维度)。
/// 维度仅用于启动日志和校验，实际推理维度由模型 config.json 决定。
const MODELS: &[(&str, &str, usize)] = &[
    ("bge-small-zh-v1.5", "BAAI/bge-small-zh-v1.5", 512),
    ("bge-small-zh", "BAAI/bge-small-zh-v1.5", 512),
    ("bge-large-zh-v1.5", "BAAI/bge-large-zh-v1.5", 1024),
    ("bge-large-zh", "BAAI/bge-large-zh-v1.5", 1024),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", 384),
    ("bge-small-en", "BAAI/bge-small-en-v1.5", 384),
    ("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", 768),
    ("bge-base-en", "BAAI/bge-base-en-v1.5", 768),
    ("multilingual-e5-small", "intfloat/multilingual-e5-small", 384),
    ("multilingual-e5-base", "intfloat/multilingual-e5-base", 768),
    ("multilingual-e5-large", "intfloat/multilingual-e5-large", 1024),
];

fn pick_model(name: &str) -> Result<(&'static str, usize)> {
    let n = name.trim().to_ascii_lowercase();
    if n.is_empty() {
        // 默认 bge-small-zh-v1.5（与 fastembed 时代保持一致）。
        return Ok(("BAAI/bge-small-zh-v1.5", 512));
    }
    MODELS
        .iter()
        .find(|(alias, _, _)| *alias == n)
        .map(|(_, repo, dim)| (*repo, *dim))
        .ok_or_else(|| {
            let available: Vec<&str> = MODELS.iter().map(|(a, _, _)| *a).collect();
            anyhow!(
                "未知 embeddings.model '{n}'；可用：{}",
                available.join(" | ")
            )
        })
}

/// 已加载的模型 + 分词器（受 Mutex 保护）。
struct Inner {
    model: BertModel,
    tokenizer: Tokenizer,
}

/// 本地 embedding 推理器（进程单例）。
pub struct LocalEmbedder {
    inner: Mutex<Inner>,
    /// 向量维度（加载时确定，供调用方校验/调试）。
    pub dim: usize,
}

static INSTANCE: OnceLock<Result<LocalEmbedder>> = OnceLock::new();

impl LocalEmbedder {
    /// 获取进程级单例。首次调用会触发模型加载（同步、可能下载 ~100MB 模型）。
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
        let (repo_id, expected_dim) = pick_model(model_name)?;

        // 构造 HF API client。from_env() 自动读 HF_HOME / HF_ENDPOINT / HF_TOKEN；
        // 若配置了 cache_dir 则覆盖。
        let mut builder = ApiBuilder::from_env();
        if let Some(dir) = cache_dir.filter(|s| !s.trim().is_empty()) {
            builder = builder.with_cache_dir(PathBuf::from(dir));
        }
        let api = builder.build()?;

        tracing::info!("正在加载本地 embedding 模型 {repo_id}（首次需下载权重）...");
        let api_repo = api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));
        let config_path = api_repo.get("config.json")?;
        let tokenizer_path = api_repo.get("tokenizer.json")?;
        let weights_path = api_repo.get("model.safetensors")?;

        // Config 反序列化（BGE 的 config.json 是标准 BERT config）。
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let dim = config.hidden_size;
        if dim != expected_dim {
            tracing::warn!(
                "模型 {repo_id} 实际维度 {dim} 与预期 {expected_dim} 不符，按实际 {dim} 处理"
            );
        }

        // 加载权重（mmap safetensors，避免读整个文件到内存）。
        // SAFETY: 标准用法，candle 官方示例 bert/main.rs 同款调用。
        // 权重来自 HuggingFace 官方 repo（BAAI / intfloat），可信。
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DTYPE, &device)?
        };
        let model = BertModel::load(vb, &config)?;

        // 分词器：启用 BatchLongest padding，批量推理时短句补齐到 batch 内最长。
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        tracing::info!(
            "本地 embedding 模型加载完成：{repo_id}（dim={dim}）。后续推理离线进行。"
        );

        Ok(LocalEmbedder {
            inner: Mutex::new(Inner { model, tokenizer }),
            dim,
        })
    }

    /// 批量 embedding（核心实现）。
    ///
    /// 仅可通过 `global()` 返回的 `&'static Self` 调用，这样 `spawn_blocking`
    /// 闭包能持有引用。
    pub async fn embed_batch(
        self: &'static Self,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>> {
        tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
            let inner = self.inner.lock().expect("LocalEmbedder mutex poisoned");
            let embeddings = forward_batch(&inner, texts)?;
            Ok(embeddings)
        })
        .await
        .map_err(|e| anyhow!("embed_batch 任务失败: {e}"))?
    }

    /// 单条文本 embedding。
    pub async fn embed(self: &'static Self, text: String) -> Result<Vec<f32>> {
        let mut out = self.embed_batch(vec![text]).await?;
        out.pop().ok_or_else(|| anyhow!("embedding 返回空"))
    }
}

/// 对一批文本跑前向 + mean pooling + L2 normalize。
fn forward_batch(inner: &Inner, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let device = &inner.model.device;

    // 分词：encode_batch 会按 BatchLongest 自动 padding 对齐。
    let encodings = inner
        .tokenizer
        .encode_batch(texts, true)
        .map_err(|e| anyhow!("分词失败: {e}"))?;

    let token_ids = encodings
        .iter()
        .map(|enc| {
            let ids = enc.get_ids().to_vec();
            Ok::<_, anyhow::Error>(Tensor::new(ids.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let attention_mask = encodings
        .iter()
        .map(|enc| {
            let mask = enc.get_attention_mask().to_vec();
            Ok::<_, anyhow::Error>(Tensor::new(mask.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;
    let token_type_ids = token_ids.zeros_like()?;

    // BERT 前向：输出 shape = [batch, seq_len, hidden_size]。
    let embeddings = inner
        .model
        .forward(&token_ids, &token_type_ids, Some(&attention_mask))
        .map_err(|e| anyhow!("模型前向失败: {e}"))?;

    // Attention-masked mean pooling：
    // 把 padding 位置的 embedding 清零后求和，除以 mask 计数，得到句向量。
    // 这与 sentence-transformers 的实现数值等价。
    let mask_f32 = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let sum_mask = mask_f32.sum(1)?;
    let sum_embeddings = embeddings.broadcast_mul(&mask_f32)?.sum(1)?;
    let pooled = sum_embeddings.broadcast_div(&sum_mask)?;

    // L2 归一化（BGE / E5 都要求归一化后做点积 = cosine）。
    let normalized = pooled.broadcast_div(&pooled.sqr()?.sum_keepdim(1)?.sqrt()?)?;

    // 取出为 Vec<Vec<f32>>：shape [batch, hidden]。
    let batch_size = normalized.dim(0)?;
    let mut out = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let row = normalized.get(i)?;
        let vec_f32 = row.to_vec1::<f32>()?;
        out.push(vec_f32);
    }
    Ok(out)
}
