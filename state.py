"""state.py —— 状态的数据结构定义、校验与持久化（V1.4：SQLite 取代本地 JSON 文件）。

V1.4：废弃本地 JSON 文件读写；持久化经 database.py 存 agent.db（user_states 表）。
本文件保留：default_state / validate / build_state_from_alignment（数据形状），
以及 load_state(session_id) / save_state(state, session_id)（持久化接口）。

V1.3（schema v3，通用学习教练）：
- config.daily_total：用户每日可支配【总时长】（学习+休息，分钟）；
  生成清单时自动核算休息耗时，超上限就删减任务（时间总铁则）。
- config.break_minutes：每段番茄后的固定休息（5-10 分钟，默认 5）。
- config.focus_duration：单次专注上限，默认 25 分钟（番茄小块）。
- flow.mode：plan/chat（闲聊模式随时切换）；flow.tier：三档方案偏好；
  flow.last_calibration：每 3-5 天主动校准模块权重的最后日期。
"""

from __future__ import annotations

import json
from datetime import date, datetime

import database

SCHEMA_VERSION = 3

# V1.3 配置默认值（由 LLM 从用户大白话里提取，提取不到时用这些）
DEFAULT_FOCUS_MINUTES = 25      # 单次专注极限（番茄块上限）
DEFAULT_TOTAL_MINUTES = 120     # 每日可支配总时长（含休息）
DEFAULT_BREAK_MINUTES = 5       # 每段番茄后的固定休息（5-10 分钟区间取 5 核算）
CAPACITY_STUDENT = 180          # LLM 提取时的参考：学生党默认总时长
CAPACITY_WORKER = 120           # LLM 提取时的参考：上班族默认总时长


def default_state(deadline: str = "", life_profile: str = "",
                  daily_total: int | None = None,
                  start_date: str | None = None) -> dict:
    """创建一个空的初始状态。materials / knowledge_points 由调用方填入。"""
    start = start_date or date.today().isoformat()
    return {
        "version": SCHEMA_VERSION,
        "meta": {
            "deadline": deadline,          # 截止日（考试/交付/完成日期），ISO "YYYY-MM-DD"
            "life_profile": life_profile,  # 自由文本："大三课多"/"打工人只能晚上学"等
            "start_date": start,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "materials": {},                 # 资源类型名 → 内容；类型名由用户自定义
        "knowledge_points": [],          # 对齐表的落地形态：知识点 × 条目
        "schedule": {},                  # "YYYY-MM-DD" → {"blocks": [...]}
        "config": {
            "focus_duration": DEFAULT_FOCUS_MINUTES,
            "daily_total": int(daily_total or DEFAULT_TOTAL_MINUTES),  # 含休息的总时长
            "break_minutes": DEFAULT_BREAK_MINUTES,
            "weekend_boost": False,
            "history": [],               # 配置变更记录（focus/total 调整）
        },
        "history": [],                   # 每日事件：report / config / cut / misaligned ...
        "warnings": [],                  # 预警记录（含砍内容提案与决策状态）
        "chat": [],                      # 对话框消息流：[{"role": "agent"/"user", "text", "time"}]
        "flow": {
            "phase": "new",              # 会话阶段机：new / await_confirm / daily / await_cut
            "asked": [],                 # 已追问事项（防复读机）
            "mode": "plan",              # plan / chat（闲聊模式）
            "tier": "standard",          # 三档方案偏好：standard / minimal / rest
            "last_calibration": None,    # 上次模块权重校准日期（每 3-5 天一次）
        },
    }


def load_state(session_id: str) -> dict:
    """按会话从数据库读取完整状态；会话不存在抛 KeyError。"""
    raw = database.load_state(session_id)
    if raw is None:
        raise KeyError(f"会话不存在：{session_id}")
    state = json.loads(raw)
    validate(state)
    return state


def save_state(state: dict, session_id: str) -> None:
    """校验后整条序列化为 JSON 存入数据库（覆盖式 upsert，原子事务）。"""
    validate(state)
    database.save_state(session_id, json.dumps(state, ensure_ascii=False))


def validate(state: dict) -> None:
    """最小结构校验：缺关键字段直接报错，不猜。"""
    required = ["version", "meta", "materials", "knowledge_points",
                "schedule", "config", "history", "warnings"]
    missing = [k for k in required if k not in state]
    if missing:
        raise ValueError(f"state.json 缺少字段：{missing}")
    if state["version"] != SCHEMA_VERSION:
        raise ValueError(f"state.json 版本不匹配：{state['version']}（期望 {SCHEMA_VERSION}）")
    if "deadline" not in state["meta"] or "life_profile" not in state["meta"]:
        raise ValueError("state.meta 必须包含 deadline 和 life_profile")
    cfg = state["config"]
    if "focus_duration" not in cfg or "daily_total" not in cfg \
            or "break_minutes" not in cfg:
        raise ValueError("state.config 必须包含 focus_duration / daily_total / break_minutes")


def build_state_from_alignment(alignment: dict, deadline: str,
                               life_profile: str = "",
                               daily_total: int | None = None,
                               start_date: str | None = None,
                               state: dict | None = None) -> dict:
    """把对齐表草稿组装成完整状态（对齐 → state 的结构胶水，无业务逻辑）。

    对齐表结构：
    {"basis": "user_materials|general_knowledge",
     "subject_name": "...",
     "resources": [...],
     "knowledge_points": [{"name", "priority", "items":
       [{"source", "role", "title", "minutes"}]}]}

    state 缺省时新建；传入时原地更新（保留 chat / flow 等既有字段），
    并按传入的容量配置重置排期参数。
    """
    if state is None:
        state = default_state(deadline, life_profile, daily_total, start_date)
    else:
        state["meta"]["deadline"] = deadline
        if life_profile:
            state["meta"]["life_profile"] = life_profile
        # 确认当天 = 排期起点（V1.2：确认后给的是今日清单）
        state["meta"]["start_date"] = start_date or date.today().isoformat()
        if daily_total:
            state["config"]["daily_total"] = int(daily_total)
        state["schedule"] = {}       # 旧日程作废，由 init_plan 重排
    # resources 归一化成字符串数组（LLM 偶发返回对象列表）
    def _rname(r):
        if isinstance(r, str):
            return r
        if isinstance(r, dict):
            return str(r.get("name") or r.get("type") or "未命名资源")
        return str(r)

    state["materials"]["resources"] = [_rname(r) for r in alignment.get("resources", [])]
    state["knowledge_points"] = []
    for i, kp in enumerate(alignment["knowledge_points"], start=1):
        kp_id = f"kp{i}"
        node = {
            "id": kp_id,
            "name": kp["name"],
            "priority": float(kp.get("priority", 1.0)),
            "status": "pending",
            "items": [],
        }
        for j, it in enumerate(kp["items"], start=1):
            minutes = max(1, int(it["minutes"]))
            node["items"].append({
                "id": f"{kp_id}-{j}",
                "source": _rname(it["source"]),
                "role": it["role"],
                "title": str(it["title"]),
                "minutes": minutes,
                "remaining_min": minutes,
                "status": "todo",
            })
        state["knowledge_points"].append(node)
    return state
