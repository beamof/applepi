# applepi

一个简洁、高效、易扩展的个人 AI Agent，用 Rust 实现。

- **对话**：基于 OpenAI 兼容接口，一套代码接 GPT / DeepSeek / 智谱 / 本地 Ollama
- **行动**：可插拔工具机制（Function Calling），自带文件读取等示例工具
- **记忆**：短期会话上下文 + 长期向量记忆（SQLite，零运维）
- **流式**：SSE 增量输出，终端逐字、Telegram 增量编辑
- **接入**：CLI 与 Telegram Bot 两个入口，共享同一套核心

---

## 快速开始

### 1. 环境要求

- Rust 1.75+（推荐 `rustup` 安装）
- 一份 OpenAI 兼容的 API Key

### 2. 配置

人设在 [`AGENTS.md`](./AGENTS.md) 里维护（启动时读一次）。其余配置在 [`config.yaml`](./config.yaml)，至少填好 `llm.api_key` 和 `llm.model`：

```yaml
llm:
  api_base: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key: sk-xxxx              # 留空则回退到环境变量 OPENAI_API_KEY

memory:
  enabled: true
  db_path: data/applepi.db
  top_k: 3                      # 每次注入 prompt 的最多相关记忆数（FTS5 BM25 排序）
```

> API Key 优先级：`config.yaml` > 环境变量 `OPENAI_API_KEY`（或 `API_KEY`）。
> 生产环境建议用环境变量，避免 key 进入 git。

### 3. 长期记忆检索（FTS5 全文检索）

长期记忆用 **SQLite FTS5** 全文检索 + BM25 评分。零 ML 模型、零网络、零 API 依赖，部署即用。rusqlite 的 `bundled` feature 默认启用 FTS5，无需额外编译。

- **写入**：每条记忆的文本存入 FTS5 虚拟表，自动建倒排索引。
- **检索**：用户消息作为 query，BM25 评分排序取 Top-K 注入 prompt。
- **中文分词**：FTS5 内置 `unicode61` 分词器对中文按字切分（"苹果派"→"苹/果/派"），短句关键词检索够用；缺点是无语义相似（"水果"匹配不到"苹果"），但记忆场景以关键词命中为主，影响可接受。
- **查询安全**：用户输入按字切分后每个字符用 `"..."` 短语引用转义，规避 FTS5 MATCH 操作符注入。

相比旧方案（embedding 模型）：

| 维度 | 旧方案（embedding） | 新方案（FTS5） |
|---|---|---|
| 部署 | 需下载 ~100MB 模型 | 零下载，即用 |
| 依赖 | candle/ort 等 ML 栈 | 仅 SQLite（已内置） |
| 语义检索 | ✅ | ❌（关键词匹配） |
| 单次查询 | ~50ms（推理） | <1ms（索引） |

#### 存量数据迁移

从旧版（向量 schema）升级时，启动会自动把旧 `memories` 表的 `text` 字段导入新 FTS5 表（向量字段丢弃），打 `tracing::info` 日志，然后升级 schema 版本到 3。文本内容不丢失，但向量索引重建为 FTS。
> 环境变量模板见 [`.env.example`](./.env.example)，复制为 `.env` 后填写。

### 3. 运行

**CLI 模式（终端逐字流式）：**

```bash
cargo run --bin cli
```

**Telegram Bot 模式：**

```bash
# 1. 找 @BotFather 创建 bot，拿到 token
# 2. 填到 config.yaml 的 telegram.bot_token，或环境变量 TELEGRAM_BOT_TOKEN
# 3. 运行
cargo run --bin bot
```

**Release 构建：**

```bash
cargo build --release --bins
# 产物：target/release/cli.exe  target/release/bot.exe
```

---

## 换模型厂商

只改 `config.yaml`，不动代码：

| 厂商 | api_base | model |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5` |

---

## 项目结构

```
applepi/
├── AGENTS.md               # 人设（系统提示）
├── config.yaml             # 模型、记忆、Telegram、MCP、Cron 配置
├── Cargo.toml
├── skills/                 # ★ 技能（每个子目录一个 SKILL.md）
└── src/
    ├── lib.rs              # 模块入口
    ├── main → 拆为 bin/    # 两个可执行入口
    ├── agent.rs            # ★ ReAct 主循环 + 记忆注入 + 事件流
    ├── llm.rs              # OpenAI 兼容客户端（流式 + 非流式）
    ├── config.rs           # YAML 配置加载
    ├── bot.rs              # Telegram 长轮询 + 流式编辑
    ├── memory/
    │   ├── short_term.rs   # 会话上下文
    │   └── long_term.rs    # ★ SQLite + 向量 + cosine 检索
    ├── mcp/                # ★ MCP 接入（HTTP/SSE 传输）
    │   ├── mod.rs          # 加载入口 + 工具合并
    │   ├── client.rs       # Streamable HTTP JSON-RPC 客户端
    │   └── tool.rs         # 远端工具 → 本地 Tool 适配器
    ├── cron/               # ★ 定时任务（仅 bot 模式）
    │   ├── mod.rs          # scheduler：cron 触发 + 独立 Agent 推送
    │   └── store.rs        # SQLite 持久化
    ├── tools/
    │   ├── mod.rs          # Tool trait + 注册表（扩展点）
    │   ├── cron.rs         # cron 管理工具（agent 调用）
    │   ├── echo.rs         # 示例工具
    │   ├── fs.rs           # read_file 工具
    │   ├── shell.rs        # shell 命令工具（白名单/黑名单）
    │   └── skill.rs        # 技能创建/运行工具
    └── bin/
        ├── cli.rs          # 终端入口
        └── bot.rs          # Telegram 入口
```

---

## 核心设计

### 工具机制（易扩展的关键）

实现 `Tool` trait 的 4 个方法，注册一行，Agent 自动具备新能力：

```rust
// src/tools/my_tool.rs
use crate::tools::Tool;
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct MyTool;

#[async_trait]
impl Tool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "做某件事" }
    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "x": { "type": "string" } } })
    }
    async fn run(&self, args: Value) -> anyhow::Result<String> {
        Ok("结果".into())
    }
}
```

在 `tools/mod.rs` 的 `default_tools()` 加一行即可启用：

```rust
let tools: Vec<Arc<dyn Tool>> = vec![
    Arc::new(echo::Echo),
    Arc::new(fs::ReadFile),
    Arc::new(my_tool::MyTool),   // ← 新增
];
```

### MCP 服务器接入（`mcp/`）

除本地实现的工具外，applepi 还能接入**远程 MCP 服务器**（[Model Context Protocol](https://modelcontextprotocol.io)，Streamable HTTP 传输）。启动时自动握手 + 拉取远端工具列表，每个远端工具被包装成本地 `Tool` 注入 Agent，**无需改一行 Agent 代码**。

在 `config.yaml` 声明即可：

```yaml
mcp_servers:
  - name: example
    url: https://mcp.example.com/mcp
    headers:                       # 可选，额外请求头
      Authorization: "Bearer xxx"
    enabled: true                  # 可选，默认 true
```

行为约定：

- **零侵入**：复用现有 `Tool` trait / `ToolMap`，Agent 主循环不变。
- **错误隔离**：单个服务器连接失败只打印 `[MCP]` 警告并跳过，不阻断启动；工具调用失败走现有 `[工具错误]` 路径喂回模型。
- **连接共享**：同一服务器的多个工具共享一个连接，避免重复握手。
- **工具名冲突**：同名工具后者覆盖前者（HashMap 语义），覆盖时打印警告。

### 定时任务 Cron（`cron/`，仅 bot 模式）

按 cron 表达式定时触发 Agent 执行预设 prompt，把回复推送到指定 Telegram chat_id。适合每日总结、定时提醒、周期巡检等场景。

**配置**（`config.yaml`）：

```yaml
cron:
  enabled: true                  # 总开关
  db_path: data/cron.db          # 持久化库（与长期记忆库分库）
  jobs:                          # 启动时种子 job（首次写入 DB，之后由 cron 工具管理）
    - name: daily_summary
      schedule: "0 9 * * *"      # 北京时间每天 9:00
      prompt: "总结今天的待办"
      chat_id: 123456789
      enabled: true
```

**运行时管理**：agent 内置 `cron` 工具，用户用自然语言对话即可让 agent 创建/查询/暂停/删除任务，无需手动发命令。例如用户说"每天 9 点提醒我站会，chat_id 是 123"，agent 会直接调用 `cron` 工具（action=add）创建任务。详见 [`AGENTS.md`](./AGENTS.md)。

**设计要点：**

- **持久化**：任务存 `data/cron.db`，重启不丢失。`config.yaml` 的 `jobs` 仅作为首次启动的种子。
- **时区**：cron 表达式按**北京时间（UTC+8）**解释。
- **错过不补**：进程停机期间错过的任务跳过，重启后从下次匹配时间继续。
- **独立 Agent**：每个任务一个独立 Agent，不与 bot 的对话 Agent 共享状态/锁。
- **热重载**：agent 通过 `cron` 工具改动 DB 后，经 watch 通道立即生效，无需重启进程。

### Shell 工具（`tools/shell.rs`，可选）

让 agent 执行 shell 命令。默认关闭，启用后受**白名单/黑名单**约束：

```yaml
shell:
  enabled: true
  allow: ["ls", "cat ", "git status", "pwd"]  # 命令前缀白名单，空则不限
  deny: []                                     # 子串黑名单，空则用内置默认
  timeout: 30
```

> ⚠️ **安全提醒**：白名单/黑名单是字符串匹配，**不是沙箱**，无法防御利用 shell 特性（管道、变量拼接等）构造的绕过。仅在可信环境、可信输入下启用；生产/多用户场景请用容器隔离或保持 `enabled: false`。

### 技能 Skills（`tools/skill.rs`，Claude Skills 风格）

技能是持久化的工作流/指令模板（`skills/<name>/SKILL.md`），让 agent 把重复性任务沉淀为可复用资产。agent 内置两个工具：

- `skill_create(name, description, content)` — 创建/更新技能文件
- `skill_use(name)` — 加载技能正文为本次任务上下文，agent 据此调用其他工具完成工作

```markdown
---
name: code-review
description: 代码评审工作流，检查风格与安全并给出结构化反馈
---
# 代码评审
...（具体指令）
```

启动时自动扫描 `skills/` 目录，把所有技能的 name + description 摘要注入人设，让 agent 感知可用技能。技能就是 Markdown 文件，人也能直接编辑。

### ReAct 主循环（`agent.rs`）

```
用户输入 → 注入长期记忆 → LLM
                          ↓
                   返回 tool_calls？──是──→ 执行工具 → 结果入历史 → 回到 LLM
                          ↓ 否
                     流式输出最终答复 → 存入长期记忆
```

最多 `MAX_TURNS`（默认 6）轮，防止死循环。

### 长期记忆（`memory/long_term.rs` + `memory/embed.rs`）

- **存储**：SQLite 单表 `memories(text, embedding, created_at)`，schema 版本由 `PRAGMA user_version` 管理
- **向量化**：本地 ONNX 推理（`LocalEmbedder` 单例），无网络往返、无 API key
- **检索**：拉取最近 2000 条 → 内存算 cosine 相似度 → Top-K
- **注入**：把命中记忆追加到 **user 消息尾部**（不污染 system，最大化 prompt 前缀缓存命中）
- **写入**：对话结束存用户原话（生产可换成 LLM 抽取要点）
- **短路**：太短（<4 字符）或纯斜杠命令的输入跳过检索

### 流式输出（`llm.rs::chat_stream`）

用 `tokio::sync::mpsc` 推送 `Delta` 事件：`Text` / `ToolCalls` / `Final`。
工具调用走非流式（拼接 arguments 更稳），纯文本走 SSE 增量。
`Agent` 统一封装成 `AgentEvent`，调用方只关心事件类型。

---

## 配置项一览

| 配置路径 | 说明 | 默认/备注 |
|---|---|---|
| `AGENTS.md` | 人设 / 系统提示 | 必填（项目根目录） |
| `llm.api_base` | OpenAI 兼容接口地址 | 必填 |
| `llm.model` | 模型名 | 必填 |
| `llm.api_key` | API Key | 留空则读环境变量 |
| `llm.prompt_cache_control` | 给 system 打 cache_control（Anthropic/OpenAI） | `false` |
| `memory.enabled` | 是否启用长期记忆 | `false` |
| `memory.db_path` | SQLite 路径 | `data/applepi.db` |
| `memory.top_k` | 注入记忆条数 | `3` |
| `telegram.bot_token` | Telegram token | 留空则读环境变量 |
| `mcp_servers` | MCP 服务器列表（HTTP/SSE） | `[]`（不接入） |
| `cron.enabled` | 是否启用定时任务 | `false` |
| `cron.db_path` | Cron 持久化库路径 | `data/cron.db` |
| `cron.jobs` | 种子任务列表（首次写入 DB） | `[]`（无） |
| `shell.enabled` | 是否启用 shell 工具 | `false` |
| `shell.allow` | 命令前缀白名单 | `[]`（不限制） |
| `shell.deny` | 子串黑名单（空用内置默认） | `[]` |
| `shell.timeout` | 执行超时（秒） | `30` |

---

## 安全提示

- `config.yaml` 和 `.env` 可能含明文密钥，请加入 `.gitignore`
- `data/applepi.db` 是用户数据，同样不要提交
- Telegram Bot 建议用环境变量传 token，而非写死配置

---

## 后续可扩展方向

- [ ] LLM 抽取事实存记忆（替代存原话）
- [ ] 多用户隔离的长期记忆（按 user_id 分库/分表）
- [ ] 知识库 RAG（文档切块入库，复用向量检索）
- [ ] Web UI（axum + SSE 端点）
- [ ] 多 Agent 协作 / 定时任务 / 语音

## License

MIT
