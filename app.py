"""app.py —— FastAPI 入口（铁序第三步）：打开就是一个对话框，串起 llm + planner。

V1.3.1（性能版）：
- POST /chat 改为 Server-Sent Events 流式响应：LLM 回复像打字机一样逐段蹦出
  （llm.stream_extract 统一一次调用完成意图判断+状态提取+回复生成，不再二次调用）。
- 每次 LLM 调用只带最小状态摘要 + 最近 3 轮对话（省 token、快首字）。
- 三档方案 / 闲聊模式 / 动态权重 / 阶段看板等 V1.3 能力不变。
- 多用户隔离：state_{session_id}.json；旧 schema 文件自动备份迁移。

运行：uvicorn app:app --reload  →  http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
import llm
import planner
import state as state_mod

BASE = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时：初始化数据库，并把旧版 state_*.json 一次性迁移进库。"""
    database.init_db()
    database.import_legacy_files()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

GREETING = (
    "你好，我是你的通用学习教练🥰\n"
    "不管是考证、考研、自学编程、语言学习、专业课复习都可以。\n\n"
    "你可以发给我教材、网课标题，或者直接告诉我三件事：\n"
    "① 目标截止时间（考试/想要完成的日期）\n"
    "② 每天能拿出的总时长（包含休息）\n"
    "③ 单次最多能专注多久\n\n"
    "我会搭建学习大纲，拆成适合你的番茄小块，同时准备标准版/保底版/休息日三套方案，"
    "累了随时休息，不内卷不施压。"
)

STARTUP_Q = {
    "exam_date": "①目标截止时间（考试/想完成的日期）？",
    "daily_total": "②每天能拿出的总时长（包含休息）？",
    "focus_duration": "③单次最多能专注多久？",
}

TIER_HINT = "\n\n状态一般回『保底』要🟡精简版；想休息就回『休息日』。"

STATIC_REPLIES = {
    "chat_mode": "好呀，切换闲聊模式～想聊什么？想继续规划的时候随时喊我。",
}

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ChatIn(BaseModel):
    message: str
    session_id: str = ""


# ---------- 状态与工具 ----------

def _safe_sid(session_id: str) -> str:
    """session_id 白名单校验（它是数据库主键，防注入/防怪异键名）。"""
    sid = (session_id or "").strip() or "default"
    if not SESSION_RE.match(sid):
        raise HTTPException(status_code=400, detail="session_id 不合法")
    return sid


def _load(session_id: str) -> dict:
    """从 SQLite 读会话状态；旧 schema/损坏数据自动重置为全新状态。"""
    sid = _safe_sid(session_id)
    try:
        s = state_mod.load_state(sid)
    except (KeyError, ValueError, json.JSONDecodeError):
        s = state_mod.default_state()
    flow = s.setdefault("flow", {"phase": "new"})
    flow.setdefault("asked", [])
    flow.setdefault("mode", "plan")
    flow.setdefault("tier", "standard")
    flow.setdefault("last_calibration", None)
    s.setdefault("chat", [])
    return s


def _save(s: dict, session_id: str) -> None:
    state_mod.save_state(s, _safe_sid(session_id))


def _say(s: dict, text: str) -> None:
    s["chat"].append({"role": "agent", "text": text, "time": date.today().isoformat()})


def _say_user(s: dict, text: str) -> None:
    s["chat"].append({"role": "user", "text": text, "time": date.today().isoformat()})


def _blocks_text(blocks: list) -> str:
    if not blocks:
        return "（没有待办任务啦）"
    return "\n".join(planner.render_blocks(blocks))


def _tomorrow() -> str:
    return (date.today() + timedelta(days=1)).isoformat()


def _reported_today(s: dict) -> bool:
    today = date.today().isoformat()
    return any(e.get("type") == "report" and e.get("date") == today
               for e in s["history"])


def _prompt(s: dict) -> str:
    """页面每次打开时，Agent 主动说的一句（不存入聊天历史）。"""
    flow = s["flow"]
    phase = flow.get("phase", "new")
    if phase == "new":
        return GREETING
    if flow.get("mode") == "chat":
        return "闲聊模式中～想继续规划的时候随时喊我。"
    if phase == "await_confirm":
        return "大纲还等你过目呢～回『行』或直接说怎么改（删除XX/增加XX/调整XX权重）。"
    if phase == "await_cut":
        return "上次的预警还等你拍板：回『砍』按建议来，或『不砍』维持全量，都听你的。"
    today = date.today().isoformat()
    board = planner.progress_board(s)
    if _reported_today(s):
        return (board + "\n\n今天已汇报，辛苦啦，安心休息～\n\n明天清单：\n"
                + _blocks_text(planner.daily_blocks(s, _tomorrow())) + TIER_HINT)
    return (board + "\n\n今晚汇报一句吧（比如『做了一半』『太难了』『没学』都行）。\n\n"
            "今天的清单：\n" + _blocks_text(planner.daily_blocks(s, today)) + TIER_HINT)


# ---------- 提取结果应用与追问 ----------

def _apply_extracted(s: dict, ex: dict) -> list:
    """V1.3 铁律：只要 LLM 提取到信息，立刻更新 state。
    返回被调整权重的模块名列表（用于回复提示）。"""
    meta, cfg = s["meta"], s["config"]
    if ex.get("exam_date"):
        meta["deadline"] = ex["exam_date"]
    if ex.get("life_profile"):
        meta["life_profile"] = ex["life_profile"]
    if ex.get("daily_total"):
        cfg["daily_total"] = ex["daily_total"]
    if ex.get("focus_duration"):
        cfg["focus_duration"] = ex["focus_duration"]
    if ex.get("weekend_boost") is not None:
        cfg["weekend_boost"] = ex["weekend_boost"]
    if ex.get("tier"):
        s["flow"]["tier"] = ex["tier"]
    adjusted = []
    for u in ex.get("priority_updates") or []:
        name, p = u["kp_name"], u["priority"]
        matched = None
        for kp in s["knowledge_points"]:
            if kp["name"] == name:
                matched = kp
                break
        if matched is None:
            for kp in s["knowledge_points"]:
                if name in kp["name"] or kp["name"] in name:
                    matched = kp
                    break
        if matched is not None:
            matched["priority"] = p
            adjusted.append((matched["name"], p))
    if adjusted:
        s["flow"]["last_calibration"] = date.today().isoformat()
    return adjusted


def _startup_has(s: dict, key: str) -> bool:
    if key == "exam_date":
        return bool(s["meta"].get("deadline"))
    if key == "daily_total":
        return bool(s["config"].get("daily_total"))
    return bool(s["config"].get("focus_duration"))


def _mark_asked(flow: dict, key: str) -> None:
    asked = flow.setdefault("asked", [])
    if key not in asked:
        asked.append(key)


def _ask_startup(s: dict, flow: dict, ex: dict) -> str:
    """只问确实还缺、且没追问过的启动信息；文案按 LLM 的 missing 来（不复读）。"""
    questions = []
    for key in ex.get("missing") or []:
        if _startup_has(s, key):
            continue
        questions.append(STARTUP_Q[key])
        _mark_asked(flow, key)
    if not questions and flow.get("phase") == "await_confirm" \
            and not _startup_has(s, "exam_date") \
            and "exam_date" not in flow.get("asked", []):
        questions.append(STARTUP_Q["exam_date"])
        _mark_asked(flow, "exam_date")
    return "\n".join(questions)


def _day_list_text(s: dict, day: str, tier: str = "standard") -> str:
    """按三档方案渲染某天清单：✅标准版 / 🟡保底精简版 / ⚫休息日版。"""
    blocks = planner.daily_blocks(s, day)
    if tier == "rest":
        return "⚫ 休息日版：今天不安排任何学习任务，纯放松，聊天、娱乐都可以～"
    if tier == "minimal":
        if not blocks:
            return "🟡 保底精简版：今天没有积压任务，直接休息也完全没问题～"
        return "🟡 保底精简版（只做 1-2 个核心番茄，其余顺延）：\n" + _blocks_text(blocks[:2])
    if not blocks:
        return "（没有待办任务啦）"
    return "✅ 标准版：\n" + _blocks_text(blocks)


# ---------- 各 action 的确定性尾巴（LLM 开场白流式在前，尾巴随后拼上） ----------

def _do_reset(s: dict, flow: dict) -> str:
    flow["phase"] = "new"
    flow["asked"] = []
    flow["mode"] = "plan"
    flow["tier"] = "standard"
    flow["last_calibration"] = None
    flow.pop("pending_alignment", None)
    flow.pop("original_text", None)
    flow.pop("warning_date", None)
    s["knowledge_points"] = []
    s["schedule"] = {}
    return "好嘞，已经清空重来啦～把新材料发我就行。"


def _do_material(s: dict, flow: dict, text: str, ex: dict) -> str:
    alignment = ex.get("alignment")
    if not alignment or not alignment.get("knowledge_points"):
        # 合并调用里对齐表缺失（LLM 偶发不执行 task）：退回独立对齐调用兜底
        # —— 只有这种降级场景才会出现第二次 LLM 调用，正常路径始终一次。
        alignment = llm.align_materials(
            text,
            exam_date=s["meta"].get("deadline") or None,
            daily_total=s["config"].get("daily_total"),
            focus_duration=s["config"].get("focus_duration"),
        )
    if not alignment or not alignment.get("knowledge_points"):
        return ("这个我还看不出你要学什么科目～把书名、考试名或博主名告诉我"
                "（比如『备考公考，用粉笔5000题』），我就能帮你搭大纲。")
    # 常识生成最多 10 个大模块（防御性兜底）
    if alignment.get("basis") == "general_knowledge" \
            and len(alignment["knowledge_points"]) > 10:
        kps = sorted(alignment["knowledge_points"],
                     key=lambda k: -float(k.get("priority", 0.0)))
        alignment["knowledge_points"] = kps[:10]
    flow["pending_alignment"] = alignment
    flow["original_text"] = text
    flow["phase"] = "await_confirm"
    generated = alignment.get("basis") == "general_knowledge"
    head = ("根据你提到的材料，我把学习大纲搭好啦（标 ? 的地方是我不太确定的）："
            if generated else
            "我把你的材料按知识点理好啦（没把握的地方标了 ?）：")
    parts = [head, llm.render_alignment(alignment)]
    if generated:
        parts.append(llm.outline_note(alignment))
    parts.append("看一遍，没问题就回『行』；想改直接说（删除XX / 增加XX / 调整XX权重）。")
    questions = _ask_startup(s, flow, ex)
    if questions:
        parts.append("顺便告诉我：" + questions)
    return "\n\n".join(parts)


def _do_alignment(s: dict, flow: dict, ex: dict, action: str) -> str:
    if action == "alignment_modify" and ex.get("alignment"):
        flow["pending_alignment"] = ex["alignment"]
    if not s["meta"].get("deadline"):
        questions = _ask_startup(s, flow, ex)
        return questions or STARTUP_Q["exam_date"]
    state_mod.build_state_from_alignment(
        flow["pending_alignment"], s["meta"]["deadline"],
        life_profile=s["meta"].get("life_profile") or "",
        daily_total=s["config"].get("daily_total"),
        state=s,
    )
    today = date.today().isoformat()
    result = planner.init_plan(s, from_date=today)
    flow["phase"] = "daily"
    cfg = s["config"]
    reply = (f"计划排好啦，截止日 {s['meta']['deadline']}。"
             f"按每天约 {cfg['daily_total']} 分钟（含休息）、单块 ≤ {cfg['focus_duration']} 分钟排的，"
             f"先从今天开始：\n\n"
             f"今天的清单：\n{_blocks_text(planner.daily_blocks(s, today))}"
             + TIER_HINT)
    board = planner.progress_board(s)
    if board:
        reply += "\n\n" + board
    if not s["meta"].get("life_profile"):
        reply += "\n\n（要是想让我更懂你的节奏，随时说一句你的日常安排～）"
    if result["warning"]:
        flow["phase"] = "await_cut"
        flow["warning_date"] = today
        reply += "\n\n" + llm.write_warning_copy(result["warning"])
    return reply


def _do_tier(s: dict, ex: dict) -> str:
    tier = ex.get("tier") or "standard"
    day = date.today().isoformat()
    if tier == "rest":
        s["flow"]["tier"] = "rest"
    return _day_list_text(s, day, tier)


def _do_config(s: dict, ex: dict) -> str:
    cfg = s["config"]
    changes = []
    if ex.get("focus_duration"):
        changes.append(f"以后每个番茄块不超过 {cfg['focus_duration']} 分钟")
    if ex.get("daily_total"):
        changes.append(f"每天总时长（含休息）约 {cfg['daily_total']} 分钟")
    if not changes:
        return "想调整什么直接说哦～比如『以后每次任务改成 30 分钟』或『每天的任务少一点』。"
    planner.reschedule(s, _tomorrow())
    reply = f"已按新节奏重排：{'；'.join(changes)}。\n\n明天清单：\n" + \
        _blocks_text(planner.daily_blocks(s, _tomorrow()))
    board = planner.progress_board(s)
    if board:
        reply += "\n\n" + board
    return reply


def _do_cut(s: dict, flow: dict, ex: dict) -> str:
    decision = ex.get("cut_decision")
    if decision is None:
        return "这个我还拿不准呢～回『砍』就按建议来，回『不砍』维持全量计划，都听你的。"
    planner.apply_cut_decision(s, flow["warning_date"], decision)
    flow["phase"] = "daily"
    flow.pop("warning_date", None)
    if decision == "accept":
        head = "好，已按建议砍掉并重排啦。别担心，省下的时间够咱们把重点吃透。"
    else:
        head = "没问题，全量保留！会有点紧，我盯着进度，真不行咱们再商量。"
    return head + "\n\n明天清单：\n" + _blocks_text(planner.daily_blocks(s, _tomorrow()))


def _do_plan_mode(s: dict, flow: dict) -> str:
    flow["mode"] = "plan"
    if flow.get("phase") != "daily":
        return ""
    return (planner.progress_board(s) + "\n\n今天的清单：\n"
            + _blocks_text(planner.daily_blocks(s, date.today().isoformat())))


def _consecutive_unfinished(s: dict) -> int:
    n = 0
    for e in reversed(s["history"]):
        if e.get("type") != "report":
            continue
        if e.get("detail", {}).get("outcome") == "done":
            break
        n += 1
    return n


def _maybe_calibration(s: dict, flow: dict, ex: dict) -> str | None:
    last = flow.get("last_calibration")
    today = date.today()
    if last:
        try:
            last_d = date.fromisoformat(str(last))
        except ValueError:
            last_d = None
        if last_d is not None and (today - last_d).days < 4:
            return None
    flow["last_calibration"] = today.isoformat()
    pending = [k for k in s["knowledge_points"]
               if k["status"] not in ("done", "cut")][:3]
    if not pending:
        return None
    names = "、".join(f"『{k['name']}』" for k in pending)
    return (f"顺便校准一下：最近 {names} 掌握得怎么样？"
            f"觉得难/卡/正确率低就告诉我，我帮你调整投入权重～")


def _do_report(s: dict, flow: dict, text: str, ex: dict) -> str:
    today = date.today().isoformat()
    parsed = ex.get("report") or {}
    parsed.setdefault("outcome", "done")
    parsed.setdefault("partial_ratio", 0.5)
    parsed.setdefault("too_hard_kps", [])
    parsed.setdefault("too_heavy", False)
    parsed.setdefault("misaligned", False)
    parsed["_text"] = text
    summary = planner.apply_report(s, today, parsed)
    parts = list(summary["summary"])
    if summary["unscheduled"]:
        warning = planner.check_warning(s, today, summary["unscheduled"])
        if warning:
            flow["phase"] = "await_cut"
            flow["warning_date"] = today
            parts.append(llm.write_warning_copy(warning))
    parts.append("明天清单：\n" + _blocks_text(planner.daily_blocks(s, _tomorrow())))
    board = planner.progress_board(s)
    if board:
        parts.append(board)
    if _consecutive_unfinished(s) >= 2:
        parts.append("连续两天没完成标准版啦，明天要不要试试 🟡保底精简版？"
                     "只做 1-2 个核心番茄就行，回『保底』我就安排。")
    calib = _maybe_calibration(s, flow, ex)
    if calib:
        parts.append(calib)
    return "\n\n".join(parts)


def _dispatch_tail(s: dict, flow: dict, text: str, ex: dict, adjusted: list) -> str:
    """各 action 的确定性尾巴（LLM 的开场白已流式输出，这里拼上清单/看板等）。"""
    action = ex.get("action", "chat")
    if action == "upload":
        return llm.UPLOAD_HINT
    if action == "reset":
        return _do_reset(s, flow)
    if action == "chat_mode":
        flow["mode"] = "chat"
        return ""
    if action == "plan_mode":
        return _do_plan_mode(s, flow)
    if action == "material":
        flow["mode"] = "plan"
        return _do_material(s, flow, text, ex)
    if action in ("alignment_confirm", "alignment_modify"):
        return _do_alignment(s, flow, ex, action)
    if action == "tier":
        return _do_tier(s, ex)
    if action == "config":
        return _do_config(s, ex)
    if action == "cut":
        return _do_cut(s, flow, ex)
    if action == "chat":
        return ""
    tail = _do_report(s, flow, text, ex)
    if adjusted:
        line = "\n".join(f"『{name}』投入权重 → {p}" for name, p in adjusted)
        tail = "已调整：\n" + line + "\n\n" + tail
    return tail


# ---------- 接口 ----------

@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html",
                        headers={"Cache-Control": "no-store"})


@app.get("/state")
def get_state(session_id: str = ""):
    s = _load(session_id)
    return {"chat": s["chat"], "prompt": _prompt(s)}


def _sse(payload: str) -> str:
    return f"data: {payload}\n\n"


def _sse_chunk(text: str) -> str:
    return _sse(json.dumps({"t": text}, ensure_ascii=False))


@app.post("/chat")
def chat(inp: ChatIn):
    text = inp.message.strip()
    _safe_sid(inp.session_id)   # 先校验 session_id，不合法直接 400
    if not text:
        return {"reply": "说点什么吧～"}
    s = _load(inp.session_id)
    _say_user(s, text)
    sid = inp.session_id
    flow = s["flow"]

    def gen():
        streamed: list = []
        ex = None
        adjusted: list = []
        tail = ""
        reply = ""
        try:
            for ev in llm.stream_extract(s, text):
                if ev[0] == "extract":
                    ex = ev[1]
                    adjusted = _apply_extracted(s, ex)   # 提取到的信息立刻落盘
                elif ev[0] == "chunk" and ev[1]:
                    streamed.append(ev[1])
                    yield _sse_chunk(ev[1])               # 打字机流式输出
            # 尾巴（清单/预警文案等）在开场白流完之后再算：
            # 预警文案的 LLM 调用不再挡住首字
            tail = _dispatch_tail(s, flow, text, ex, adjusted) if ex else ""
            streamed_text = "".join(streamed).strip()
            # 静态尾巴类 action（reset/upload）：LLM 若已说了，就不重复拼接
            if ex and ex.get("action") in ("upload", "reset") and streamed_text:
                tail = ""
            if streamed_text and tail:
                reply = streamed_text + "\n\n" + tail
                yield _sse_chunk("\n\n" + tail)
            elif tail:
                reply = tail
                yield _sse_chunk(tail)
            else:
                reply = streamed_text or llm._PULL_BACK.get(flow.get("phase", "new"),
                                                            "我在呢～")
                if not streamed_text:
                    yield _sse_chunk(reply)
            _say(s, reply)
            _save(s, sid)
            yield _sse("[DONE]")
        except GeneratorExit:
            if reply:
                _say(s, reply)
                _save(s, sid)
            raise
        except Exception as e:
            if streamed:
                note = f"\n\n（回复中途断了：{e}，可以重新发一次）"
                reply = "".join(streamed).strip() + note
                yield _sse_chunk(note)
            else:
                reply = f"哎呀，出了点小状况：{e}\n（你的进度都还在，稍等一下重试或换种说法就行～）"
                yield _sse_chunk(reply)
            _say(s, reply)
            _save(s, sid)
            yield _sse("[DONE]")

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
