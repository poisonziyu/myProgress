# myProgress —— 通用学习教练 Agent

一句话：**贴材料或一句话说目标，就能拿到按番茄拆分的每日清单**的通用学习规划 Agent。支持考证、考研、考公、语言学习、专业课、自学项目等任意学习目标；不内置任何学科规则；未完成自动合并顺延，学不完提前预警并给出砍内容建议，最终决定权永远在用户。

## 项目背景

市面上的学习计划工具要么逼用户填表单手动排期，要么用"逾期警告"制造焦虑。myProgress 反其道而行：

- **用户永不手动排期** —— 只需说三件事：目标截止时间、每天能拿出的总时长（含休息）、单次专注上限；
- **材料输入零负担** —— 粘贴目录、或只说一句"备考公考用粉笔5000题"即可，Agent 不联网、不爬虫；
- **顺延无负罪感** —— 没做完自动合并到明天，禁止催债文案；连续完不成主动推荐保底精简版；
- **提前预警不隐瞒** —— "考前学不完"在数学上成立时立即告知并给砍内容建议，永远保留"不砍"选项。

## 核心架构

```
┌────────────────────────── 浏览器（static/index.html，原生 JS 单页） ──────────────────────────┐
│                          极简对话框：消息流 + 输入框，无其他元素                               │
└────────────────┬─────────────────────────────────────────────┬──────────────────────────────┘
                 │ POST /chat（SSE 流式）                       │ GET /state（开场白+聊天记录）
┌────────────────▼─────────────────────────────────────────────▼──────────────────────────────┐
│ app.py —— FastAPI 入口                                                                      │
│  · 会话阶段机：new → await_confirm → daily → await_cut                                      │
│  · SSE 打字机：LLM 开场白先流式输出，确定性尾巴（清单/看板）随后拼接                          │
└───────┬─────────────────────────────────┬──────────────────────────────────┬───────────────┘
        │                                 │                                  │
┌───────▼──────────────┐   ┌──────────────▼───────────────┐   ┌──────────────▼──────────────┐
│ llm.py（LLM 大脑）    │   │ planner.py（确定性排程引擎）   │   │ state.py + database.py       │
│  · stream_extract：  │   │  · 番茄切块（≤专注上限）       │   │  · schema v3 定义与校验       │
│    单次调用完成       │   │  · 时间总铁则（含休息核算）    │   │  · SQLite agent.db（WAL）    │
│    意图+提取+回复     │   │  · 学→练→复盘闭环优先排课      │   │  · 整条 JSON 覆盖写（upsert） │
│  · 对齐表起草         │   │  · 顺延合并重排               │   │  · 旧 JSON 自动迁移          │
│  · 预警/砍内容文案    │   │  · 预警+贪心砍内容提案         │   └─────────────────────────────┘
│  · 失败时关键词兜底   │   │  · 三阶段/三档方案/动态权重    │
└──────────────────────┘   └──────────────────────────────────┘
```

**分工铁律**：LLM 只做"理解与表达"（对齐表、状态提取、文案），排程/切块/顺延/预警触发全部是纯 Python（`planner.py`），可离线单元测试。每条用户消息只允许一次 LLM 调用。

**晚间提醒（定时任务）**：`evening_reminder.py` 是独立 CLI，由外部定时器（如 WorkBuddy 自动化 / cron）每天 20:00 拉起，遍历库中所有会话写入晚间汇报提醒（幂等，`user_states.last_reminded_at` 防重复），与 Web 服务生命周期解耦。

## 快速开始

```bash
pip install -r requirements.txt
copy .env.example .env      # 填入 LLM_API_KEY（其余默认 DeepSeek）
python smoke_test.py        # 离线冒烟测试（无需 key）
uvicorn app:app --reload    # 打开 http://127.0.0.1:8000
```

## Demo 截图

| 页面 | 截图 |
|------|------|
| 首页对话框（新用户引导） | ![首页](docs/screenshot-home.png) |
| LLM 生成的对齐表（待确认） | ![对齐表](docs/screenshot-alignment.png) |
| 每日清单 + 进度看板 | ![每日清单](docs/screenshot-daily.png) |
| 学不完预警 + 砍内容建议 | ![预警](docs/screenshot-warning.png) |

> 截图占位：截图文件请放到 `docs/` 目录，文件名与上表一致。

## API 说明

### `GET /`
返回极简对话框页面（`static/index.html`）。

### `GET /state?session_id={sid}`
返回页面打开时渲染所需的状态。

```json
{
  "chat":  [{"role": "agent", "text": "……", "time": "2026-11-10"}],
  "prompt": "开场白（按阶段动态生成）"
}
```

### `POST /chat`
**Server-Sent Events 流式响应**（`text/event-stream`），正文像打字机逐段返回。

请求体：

```json
{ "message": "我要备考2027年3月1日的研究生英语，每天3小时", "session_id": "default" }
```

事件流格式（每行 `data: <json>`）：

```
data: {"t": "好嘞，"}          ← 逐段增量文本
data: {"t": "计划排好啦……"}
data: [DONE]                  ← 结束标记
```

- `session_id` 允许 `[A-Za-z0-9_-]{1,64}`，不合法返回 400；不同 session_id 完全隔离。
- 一次调用内部只产生一次 LLM 请求（意图判断 + 状态提取 + 回复生成）；对齐表生成仅在此调用缺失时允许一次补调。

### 数据模型（简）

- **知识点（KP）**：调度基本单位，带动态权重 `priority`；
- **任务块（block）**：每日清单中的番茄小块，`□ 任务（纯学习 X 分钟）/ └ 休息：5-10 分钟`；
- **三档方案**：✅标准版 / 🟡保底精简版 / ⚫休息日版；
- **状态存储**：SQLite `user_states` 表（`session_id` 主键 + `state_json` + `updated_at` + `last_reminded_at`），WAL 模式。

## 配置

环境变量（优先）或项目根 `.env`（模板见 `.env.example`）：

| 变量 | 必填 | 说明 |
|------|------|------|
| `LLM_API_KEY` | ✅ | OpenAI 兼容网关的 API Key（默认 DeepSeek） |
| `LLM_BASE_URL` | — | 网关地址，以 `/v1` 结尾，默认 `https://api.deepseek.com/v1` |
| `LLM_MODEL` | — | 默认 `deepseek-chat` |
| `DB_PATH` | — | SQLite 文件路径；云部署指向持久化卷路径 |

## 测试

```bash
python smoke_test.py        # 离线：排程/顺延/容量下调/预警砍内容 4 大场景
python llm_smoke_test.py    # Part A 离线结构链路；Part B 联机（需 key）
```

## 部署

见 [部署到 Render 的清单](#部署到-render-的清单)（下节）。

## 部署到 Render 的清单

1. **服务类型**：Web Service，仓库连接后 Build Command 填 `pip install -r requirements.txt`，**Start Command**：
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
2. **环境变量**（Render Dashboard → Environment）：
   | 变量 | 值 |
   |------|----|
   | `LLM_API_KEY` | 你的 DeepSeek/网关 Key（用 Dashboard 填，**不要写进代码**） |
   | `LLM_BASE_URL` | `https://api.deepseek.com/v1`（可省） |
   | `LLM_MODEL` | `deepseek-chat`（可省） |
   | `DB_PATH` | `/var/data/agent.db`（配合持久化卷） |
3. **持久化卷（必须）**：Render 免费实例磁盘是**临时的**，重启即清空；请创建 Persistent Disk 挂载到 `/var/data`，并把 `DB_PATH` 指向该目录——SQLite 文件不落盘的话，所有用户的学习计划和聊天记录会随每次部署丢失。SQLite 单写者模型与本项目"短连接 + WAL"设计在单实例下没有并发问题；若未来多实例横向扩容，需换 PostgreSQL。
4. **健康检查**：可用 `GET /` 作为 Health Check Path。
5. **定时任务（可选）**：`evening_reminder.py` 若要在 Render 上跑，用 Cron Job 服务，Start Command `python evening_reminder.py`，且挂载**同一个**持久化卷。

## 许可证

仅供学习交流使用。
