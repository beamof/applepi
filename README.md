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

embeddings:
  model: text-embedding-3-small # 长期记忆用，本地可换 bge-m3

memory:
  enabled: true
  db_path: data/applepi.db
  top_k: 3
```

> API Key 优先级：`config.yaml` > 环境变量 `OPENAI_API_KEY`（或 `API_KEY`）。
> 生产环境建议用环境变量，避免 key 进入 git。
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
    │   └── fs.rs           # read_file 工具
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

### ReAct 主循环（`agent.rs`）

```
用户输入 → 注入长期记忆 → LLM
                          ↓
                   返回 tool_calls？──是──→ 执行工具 → 结果入历史 → 回到 LLM
                          ↓ 否
                     流式输出最终答复 → 存入长期记忆
```

最多 `MAX_TURNS`（默认 6）轮，防止死循环。

### 长期记忆（`memory/long_term.rs`）

- **存储**：SQLite 单表 `memories(text, embedding, created_at)`
- **向量化**：调 `/embeddings` 接口
- **检索**：拉取最近 2000 条 → 内存算 cosine 相似度 → Top-K
- **注入**：每次对话前把相关记忆追加到 system prompt 末尾（不覆盖人设）
- **写入**：对话结束存用户原话（生产可换成 LLM 抽取要点）

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
| `embeddings.model` | 向量模型 | `text-embedding-3-small` |
| `embeddings.api_base` | 向量接口地址 | 留空复用 `llm.api_base` |
| `memory.enabled` | 是否启用长期记忆 | `false` |
| `memory.db_path` | SQLite 路径 | `data/applepi.db` |
| `memory.top_k` | 注入记忆条数 | `3` |
| `telegram.bot_token` | Telegram token | 留空则读环境变量 |
| `mcp_servers` | MCP 服务器列表（HTTP/SSE） | `[]`（不接入） |
| `cron.enabled` | 是否启用定时任务 | `false` |
| `cron.db_path` | Cron 持久化库路径 | `data/cron.db` |
| `cron.jobs` | 种子任务列表（首次写入 DB） | `[]`（无） |

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
