"""llm.py —— 所有 LLM 调用（铁律 14：LLM 是产品的大脑，排程是确定性代码）。

传输协议：OpenAI 兼容 Chat Completions（当前接 DeepSeek 或任意网关），
用 openai 库 + 你自己的 Base URL 和 API Key，不依赖 Anthropic SDK。

V1.3.1 能力（每次调用都注入当前时间；见 _chat / _chat_stream）：
1. stream_extract()       核心：统一流式调用。一次 LLM 请求完成【意图判断 + 状态提取
                          + 回复生成】：输出 JSON 头 + ===REPLY=== + 回复正文，
                          JSON 静默解析、正文逐段流式产出（打字机效果）。
                          贴材料时对齐表生成也并入同一次调用（task 随 payload 下发）。
2. align_materials()      对齐表起草（保留接口：llm_smoke_test 与降级路径用）
3. parse_report()         每日一句话解析（保留为兼容接口/兜底）
4. write_warning_copy()   预警与砍内容文案（LLM 失败时用确定性文案兜底）

配置（环境变量或项目根目录 .env，模板见 .env.example；环境变量优先于 .env）：
  LLM_API_KEY    必填
  LLM_BASE_URL   必填，网关地址以 /v1 结尾，如 https://api.deepseek.com/v1
  LLM_MODEL      默认 deepseek-chat
  兼容 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL 作为备选
"""

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

# openai 库会通过 logging 记录网关返回的错误（可能含中文）；
# Windows 管道默认 ascii 编码会导致 UnicodeEncodeError，先统一成 UTF-8。
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from openai import OpenAI, AuthenticationError
except ImportError:  # 未安装 openai 库时给友好报错，而不是 import 即崩
    OpenAI = AuthenticationError = None

_ENV_FILE = Path(__file__).parent / ".env"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"   # 没填 Base URL 时默认走 DeepSeek

_client = None


def _load_dotenv():
    """极简 .env 加载：KEY=VALUE 每行一条；不覆盖已存在的环境变量。"""
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name, *fallbacks):
    for n in (name, *fallbacks):
        if os.environ.get(n):
            return os.environ[n]
    return ""


def _get_client():
    global _client
    if OpenAI is None:
        raise RuntimeError("未安装 openai 库：请先 pip install openai")
    if _client is not None:
        return _client
    api_key = _env("LLM_API_KEY", "OPENAI_API_KEY")
    base_url = _env("LLM_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_BASE_URL
    if not api_key:
        raise RuntimeError(
            "缺少 LLM_API_KEY。请在项目根目录 .env 里填写：\n"
            "  LLM_API_KEY=你的key\n"
            "  LLM_BASE_URL=https://api.deepseek.com/v1   # 可选，默认就是这个\n"
            "  LLM_MODEL=deepseek-chat\n"
            "（模板见 .env.example；或设置同名环境变量）"
        )
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError:
        raise RuntimeError(
            "LLM_API_KEY 含非 ASCII 字符——看起来还是 .env 里的占位符没替换。\n"
            "请打开项目根目录的 .env，把 LLM_API_KEY 换成网关签发的真实 API Key。"
        )
    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def _now_cn() -> str:
    """当前时间的口语化描述（V1.2 铁律：每次 LLM 调用都必须注入当前时间）。"""
    now = datetime.now()
    week = "一二三四五六日"[now.weekday()]
    return f"{now.year} 年 {now.month} 月 {now.day} 日 周{week}"


def _chat(system: str, user: str, max_tokens: int = 4096) -> str:
    client = _get_client()
    model = _env("LLM_MODEL", "OPENAI_MODEL") or DEFAULT_MODEL
    base_url = _env("LLM_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_BASE_URL
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                # V1.2：每次调用统一注入当前时间（倒计时/年份推断的依据）
                {"role": "system", "content": system + "\n\n当前时间：" + _now_cn() + "。"},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
    except AuthenticationError as e:
        raise RuntimeError(
            f"401 认证失败：{e}\n"
            f"当前 base_url = {base_url}\n"
            "请确认 LLM_API_KEY 是【该网关】签发的 API Key（比如百炼控制台的 key），"
            "而不是其他服务（如 OpenAI 官方）的 key。"
        ) from e
    return resp.choices[0].message.content or ""


def _as_dict(raw: str) -> dict:
    """把 LLM 的 JSON 文本解析成 dict，容忍前后的杂音。"""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"LLM 未返回 JSON：{text[:200]}")
    return json.loads(text[start:end + 1])


# ---------- 0. 人设与意图路由（V1.1） ----------

PERSONA = (
    "你的人设：一个懂心理学、绝不制造焦虑、温柔但靠谱的【通用学习教练】。\n"
    "学生、宝妈、在职人都能用；支持考证、考研、考公、语言、专业课、自学项目等任意学习目标；"
    "不内置任何固定学科规则，学科策略由用户给的材料动态生成。\n"
    "说话风格：口语化，自然使用语气词（如『没问题』『辛苦啦』『别急』『好嘞』）。\n"
    "交互底线：禁止说教、施压、道德绑架、PUA；禁用催债式措辞（『逾期』『落后了』『你还欠』）。"
    "用户说不想学/想玩/状态差：允许休息，主动提供保底方案，绝不无脑顺延任务。\n"
    "角色边界：用户想闲聊就好好陪聊，收口温和，不强行拽回学习；用户随时可以切回规划。\n"
)

REPLY_DELIM = "===REPLY==="

EXTRACT_SYSTEM = PERSONA + """
你是学习计划 Agent 的状态提取器与路由器。系统给你一个 JSON：now=当前时间、state=状态摘要（阶段/截止日/每日总时长/专注上限/模块名/今天清单/模式）、asked=已追问事项、recent=最近对话、text=用户刚说的话、task=可能出现的补充任务。

输出严格两段，中间用单独一行 ===REPLY=== 分隔：
第一段：一个 JSON 对象（结构化提取；null 的字段直接省略不写，尽量短）：
{"action":"...","exam_date":null,"life_profile":null,"daily_total":null,"focus_duration":null,"weekend_boost":false,"tier":null,"priority_updates":[],"alignment":null,"report":null,"cut_decision":null,"missing":[]}
第二段：reply 正文，直接对用户说的话。口语化、温柔、简短（≤60 字；action=chat 时是完整回复，要陪聊、收口温和不拽回学习；其他 action 只写一句开场白，禁止重复大纲/清单内容；action=reset/upload 时输出空）。

action：material（给材料：贴目录或一句话说书名/博主/考试）/ alignment_confirm（确认词如"行"）/ alignment_modify（改大纲，alignment 返回修改后的完整对齐表，保留 basis/subject_name）/ report（汇报今天）/ config（改配置）/ cut（回应砍内容预警）/ tier（"保底"→tier=minimal；"休息日/不学了"→rest；"标准版"→standard）/ chat_mode（要闲聊）/ plan_mode（切回规划）/ reset（重新开始）/ upload（想传图文件）/ chat（其余）
exam_date：提取截止日期转 YYYY-MM-DD；缺年份按 now 推断（月份已过→明年）；没有就省略，绝不编造。
life_profile：生活状态归纳（"大三课多"），没有省略。
daily_total：每日可支配总时长（含休息）分钟（"每天3小时"→180）；只给身份：学生→180、上班→120；没有省略。
focus_duration：单次专注上限分钟（"每次30分钟"→30），没有省略。
weekend_boost：上班族 true。
tier：仅 action=tier 填。
priority_updates：反馈模块掌握情况时填 [{"kp_name":"资料分析","priority":0.9}]（难/卡→升 0.1-0.15，熟→降 0.1；kp_name 取自 state.kp_names）；没有省略。
report：仅 action=report 填 {"outcome":"done|partial|skipped","done_blocks":[已完成块 idx],"partial_ratio":0.5,"too_hard_kps":[],"too_heavy":false,"misaligned":false}。"前两部分完成了"→done_blocks=[1,2]，拿不准就 []。today_blocks 是【今天的清单】，明天清单由系统重排，不用你算。
cut_decision：仅 action=cut 填（"不砍"→reject）。
alignment：仅 action=alignment_modify 或 task 要求时填完整对齐表。
missing：启动还缺的信息，取值限于 ["exam_date","daily_total","focus_duration"]，排除 asked 已问过的；没有省略。

只输出上面两段，不要输出任何其他文字。"""

_VALID_ACTIONS = {"chat", "material", "alignment_confirm", "alignment_modify",
                  "report", "config", "cut", "reset", "upload",
                  "tier", "chat_mode", "plan_mode"}

# 仅 phase=new 时随 payload 下发（合并"贴材料→对齐"为同一调用，不常驻 system）
MATERIAL_TASK = (
    "若 action=material，在 alignment 字段输出完整对齐表："
    '{"basis":"user_materials|general_knowledge","subject_name":"...","resources":["粉笔5000题"],'
    '"knowledge_points":[{"name":"...","priority":0.9,"confident":true,"items":[{"source":"...",'
    '"role":"learn|practice|review|reference","title":"...","minutes":25,"confident":true}]}]}。'
    "完整目录→静态对齐：resources 识别资源类型，title 只准用原文，禁止编造页码/题号，没把握的 confident=false；"
    "一句话（书名/博主/考试）→常识生成通用大模块（basis=general_knowledge，subject_name 填名称）："
    "≤10 个知识点、每点 2-3 条目按学→练→复盘闭环（learn 听课/practice 刷题/review 复盘/reference 背诵查漏）、"
    "minutes 每块 20-40 且 ≤ focus_duration、生成内容 confident 一律 false；"
    "总学习量 ≤ 剩余天数×(daily_total−休息开销)×0.8（无截止日按 4-8 周估）；识别不出科目时 knowledge_points=[]。"
)


_MATERIAL_HINT_WORDS = (
    "备考", "考研", "考公", "公务员", "考试", "红宝书", "粉笔", "题库",
    "网课", "教材", "真题", "博主", "讲义", "单词", "刷题", "学习",
)


def _norm_date(v) -> str | None:
    if not v or not isinstance(v, str):
        return None
    s = v.strip().replace("/", "-")
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        return None


def _norm_minutes(v) -> int | None:
    try:
        n = int(v)
        return n if 10 <= n <= 600 else None
    except (TypeError, ValueError):
        return None


def _normalize_extract(ex: dict) -> dict:
    """把 LLM 输出规范成固定形状（缺省值兜底，类型强转），下游放心消费。"""
    report = ex.get("report")
    if isinstance(report, dict):
        report = dict(report)
        report["done_blocks"] = [
            int(i) for i in report.get("done_blocks") or []
            if isinstance(i, (int, float)) and 1 <= int(i) <= 20
        ]
        try:
            report["partial_ratio"] = max(0.0, min(1.0, float(report.get("partial_ratio", 0.5))))
        except (TypeError, ValueError):
            report["partial_ratio"] = 0.5
    updates = []
    for u in (ex.get("priority_updates") or []):
        if not isinstance(u, dict):
            continue
        name = str(u.get("kp_name") or u.get("kp") or "").strip()
        try:
            p = max(0.1, min(1.0, float(u.get("priority"))))
        except (TypeError, ValueError):
            continue
        if name:
            updates.append({"kp_name": name, "priority": p})
    return {
        "action": ex.get("action") if ex.get("action") in _VALID_ACTIONS else "chat",
        "exam_date": _norm_date(ex.get("exam_date")),
        "life_profile": str(ex.get("life_profile") or "").strip() or None,
        "daily_total": _norm_minutes(ex.get("daily_total")),
        "weekend_boost": ex.get("weekend_boost") if isinstance(ex.get("weekend_boost"), bool) else None,
        "focus_duration": _norm_minutes(ex.get("focus_duration")),
        "tier": ex.get("tier") if ex.get("tier") in ("standard", "minimal", "rest") else None,
        "priority_updates": updates,
        "alignment": ex.get("alignment") if isinstance(ex.get("alignment"), dict) else None,
        "report": report if isinstance(report, dict) else None,
        "cut_decision": ex.get("cut_decision") if ex.get("cut_decision") in ("accept", "reject") else None,
        "missing": [m for m in (ex.get("missing") or [])
                    if m in ("exam_date", "daily_total", "focus_duration")],
        "note": str(ex.get("note") or "").strip(),
    }


def _state_summary(state: dict) -> dict:
    """给提取器的状态摘要（V1.3.1 性能版：只保留提取与路由必需的最小字段）。"""
    flow = state.get("flow", {})
    meta = state.get("meta", {})
    cfg = state.get("config", {})
    today = date.today().isoformat()
    blocks = [
        {"idx": i, "kp_id": b["kp_id"], "role": b["role"],
         "title": b["title"], "minutes": b["minutes"]}
        for i, b in enumerate(state.get("schedule", {}).get(today, {}).get("blocks", []), start=1)
    ]
    return {
        "phase": flow.get("phase", "new"),
        "exam_date": meta.get("deadline") or None,
        "daily_total": cfg.get("daily_total"),
        "focus_duration": cfg.get("focus_duration"),
        "mode": flow.get("mode", "plan"),
        "kp_names": [k.get("name") for k in state.get("knowledge_points", [])][:10],
        "today_blocks": blocks,
        "has_pending_warning": flow.get("phase") == "await_cut",
    }


def _fallback_extract(state: dict, text: str) -> dict:
    """LLM 挂掉时的确定性兜底（仅应急；主路径永远是 LLM 提取）。"""
    from planner import parse_report as keyword_parse_report
    flow = state.get("flow", {})
    phase = flow.get("phase", "new")
    base = {"action": "chat", "exam_date": None, "life_profile": None,
            "daily_total": None, "weekend_boost": None, "focus_duration": None,
            "tier": None, "priority_updates": [], "alignment": None,
            "report": None, "cut_decision": None, "missing": [], "note": ""}
    if any(w in text for w in ("重新开始", "重来", "从头开始")):
        return {**base, "action": "reset"}
    if any(w in text for w in ("上传", "发图", "传图", "图片", "照片")):
        return {**base, "action": "upload"}
    if any(w in text for w in ("休息日", "不学了", "今天不学")):
        return {**base, "action": "tier", "tier": "rest"}
    if any(w in text for w in ("保底", "精简版")):
        return {**base, "action": "tier", "tier": "minimal"}
    if phase == "new":
        if len(text) >= 20 or any(w in text for w in _MATERIAL_HINT_WORDS):
            return {**base, "action": "material"}
        return base
    if phase == "await_confirm":
        return {**base, "action": "alignment_confirm"}
    if phase == "await_cut":
        d = None
        if any(w in text for w in ("不砍", "保留", "别砍", "维持全量")):
            d = "reject"
        elif any(w in text for w in ("砍", "同意", "确认", "按建议", "按你说的")):
            d = "accept"
        return {**base, "action": "cut", "cut_decision": d}
    return {**base, "action": "report", "report": keyword_parse_report(text)}


_PULL_BACK = {
    "new": "回到正题～把材料目录或书名/博主名发我，我帮你搭大纲。",
    "await_confirm": "回到正题～大纲还等你过目：回『行』，或直接说怎么改。",
    "daily": "回到正题～晚上用一句话汇报今天就行（比如『做了一半』『没学』）。",
    "await_cut": "回到正题～上一个预警还等你拍板：回『砍』或『不砍』，都听你的。",
}

UPLOAD_HINT = "目前只支持文本粘贴哦～把材料目录直接复制成文字发给我就行，图片和文件我暂时看不了。"


def _chat_stream(system: str, user: str, max_tokens: int = 2048):
    """流式调用：逐段产出 content 文本（V1.3.1 打字机输出）。"""
    client = _get_client()
    model = _env("LLM_MODEL", "OPENAI_MODEL") or DEFAULT_MODEL
    base_url = _env("LLM_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_BASE_URL
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system + "\n\n当前时间：" + _now_cn() + "。"},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            stream=True,
        )
    except AuthenticationError as e:
        raise RuntimeError(
            f"401 认证失败：{e}\n"
            f"当前 base_url = {base_url}\n"
            "请确认 LLM_API_KEY 是【该网关】签发的 API Key（比如百炼控制台的 key），"
            "而不是其他服务（如 OpenAI 官方）的 key。"
        ) from e
    for chunk in resp:
        piece = chunk.choices[0].delta.content or ""
        if piece:
            yield piece


def _recent_chat(state: dict, n: int = 3) -> list:
    """最近 n 轮对话（V1.3.1：历史只传最近 3 轮，且每条截断，省 token）。"""
    out = []
    for m in state.get("chat", [])[-n:]:
        out.append({"role": m.get("role"), "text": str(m.get("text", ""))[:150]})
    return out


def _parse_head(raw: str) -> dict | None:
    try:
        return _as_dict(raw)
    except Exception:
        return None


def _clean_alignment(ex: dict) -> dict:
    if ex.get("alignment") is not None and \
            "knowledge_points" not in ex["alignment"]:
        ex["alignment"] = None
    return ex


def stream_extract(state: dict, text: str):
    """V1.3.1 统一流式调用：一次 LLM 请求完成【意图判断 + 状态提取 + 回复生成】。

    输出协议：第一段 JSON 头（结构化提取，静默缓冲）→ 单独一行 ===REPLY=== →
    第二段回复正文（逐段产出）。
    产出事件序列：("extract", dict) → ("chunk", str) × N → ("end", None)。
    格式异常自动降级（整段当 JSON 解析 / 关键词兜底），晚间循环不中断。"""
    flow = state.get("flow", {})
    phase = flow.get("phase", "new")
    payload = {
        "now": _now_cn(),
        "state": _state_summary(state),
        "asked": flow.get("asked", []),
        "recent": _recent_chat(state),
        "text": text[:2000],
    }
    # 贴材料时把对齐任务一并下发：合并"意图判断 + 对齐生成"为同一次调用
    if phase == "new":
        payload["task"] = MATERIAL_TASK
    pending = flow.get("pending_alignment")
    if isinstance(pending, dict) and pending.get("knowledge_points"):
        payload["pending_alignment"] = pending
    # 日常消息 2048 足够（JSON 头 + 短回复）；建大纲/改大纲才需要大额度
    max_tokens = 8192 if phase in ("new", "await_confirm") else 2048

    buffer = ""
    head_done = False
    ex_out = None
    try:
        for piece in _chat_stream(EXTRACT_SYSTEM,
                                  json.dumps(payload, ensure_ascii=False),
                                  max_tokens):
            buffer += piece
            if not head_done:
                idx = buffer.find(REPLY_DELIM)
                if idx != -1:
                    head, tail = buffer[:idx], buffer[idx + len(REPLY_DELIM):]
                    ex_out = _clean_alignment(
                        _normalize_extract(_parse_head(head) or _fallback_extract(state, text))
                    )
                    yield ("extract", ex_out)
                    head_done = True
                    if tail:
                        yield ("chunk", tail.lstrip("\n"))
            else:
                yield ("chunk", piece)
        if not head_done:
            # 没有分隔符：整段按旧格式 JSON 解析；失败则关键词兜底
            ex_out = _clean_alignment(
                _normalize_extract(_parse_head(buffer) or _fallback_extract(state, text))
            )
            yield ("extract", ex_out)
            yield ("chunk", "")
        yield ("end", None)
    except Exception:
        if not head_done:
            yield ("extract", _fallback_extract(state, text))
            yield ("chunk", "")
        yield ("end", None)


# ---------- 1. 对齐表起草（铁律 4） ----------

ALIGNMENT_SYSTEM = """你是学习计划 Agent 的材料对齐器。

用户会发来学习材料信息，先判断属于哪种情况：
A. 粘贴了完整的材料目录（视频合集标题列表、教材章节目录、题库模块列表等）
B. 只给了一句话（书名/博主/考试名，如"我要看考研红宝书，顺便看B站某博主的网课""备考公考，用粉笔5000题"）

只输出一个 JSON 对象：
{
  "basis": "user_materials | general_knowledge",
  "subject_name": "考研红宝书 / 公考（B 情况必填，书名或考试名，不加书名号）",
  "resources": ["视频源", "主教材", "习题库"],
  "knowledge_points": [
    {
      "name": "资料分析-增长率",
      "priority": 0.9,
      "confident": true,
      "items": [
        {"source": "视频源", "role": "learn", "title": "P1 增长率的概念与计算", "minutes": 25, "confident": true}
      ]
    }
  ]
}

情况 A 规则（静态对齐）：
1. resources：识别用户手里有哪些资源类型。
2. 提炼公共的"知识点轴"（通常题库的模块结构或教材的章节是最干净的本体），每个知识点一行。
3. 条目 role 四选一：learn（听课学习，如看视频/读教材正文）、practice（刷题实操，如刷题/动手做）、review（复盘整理，如错题复盘/整理笔记）、reference（背诵查漏，如背诵/查漏补缺）。同一知识点尽量学→练→复盘闭环。
4. minutes 按常识估算，每块不超过用户单次专注上限（输入会给 focus_duration），不确定就估 20-30。
5. title 只准用用户粘贴的原文描述，禁止编造页码、题号或任何用户没给的信息。
6. 没把握的映射把 confident 标为 false（会显示"?"让用户确认）；映射不上的条目可以丢弃。

情况 B 规则（常识生成）：
1. basis="general_knowledge"，subject_name 填书名/考试名（不加书名号）。
2. 根据这本书/这个考试/这位博主的常规知识体系，生成通用的"大模块"知识框架。示例：公考 = 言语理解、数量关系、判断推理、资料分析、常识判断；考研英语 = 词汇、长难句、阅读、完形、翻译、写作。
3. knowledge_points 最多 10 个，只抓大模块，不要细分小节、不要出现第 N 章级别的颗粒度。
4. 每个知识点 2-3 个条目，role 按 learn/practice/review/reference 合理搭配（闭环优先）；title 用通用模块名（如"高频词根词缀""资料分析-增长率"），禁止编造具体页码、具体题号、具体视频集数。
5. minutes 每块估 20-40 分钟，不超过单次专注上限。
6. 所有 kp 和条目 confident 一律标 false（这些是建议，等待用户确认或修改）。
7. resources 取用户提到的材料名（如"红宝书""B站网课""粉笔5000题"）。

体量推算规则（V1.3 时间总铁则）：
- 输入会给出 today、exam_date（可能为空）、daily_total（每日总时长，含休息）、
  focus_duration（单次专注上限）。
- 若 exam_date 已知：自己算出剩余天数，总学习量（所有条目 minutes 之和）控制在
  【剩余天数 × (daily_total - 休息开销) × 0.8】以内，知识点数量与条目也相应收敛
  （时间越紧，模块越少）。休息开销按每块约 5-10 分钟估。
- 若 exam_date 未知：按 4-8 周的可消化量级生成，宁少勿多。

公共规则：
1. priority：按考试分值占比或重要性的常识给 0-1（越大越优先）。
2. 不要联网搜索；不要编造用户没提供的具体信息；情况 B 实在无法识别科目时 knowledge_points 返回空数组。
3. 只输出 JSON，不要输出任何其他文字。"""


def align_materials(pasted_text: str, exam_date: str | None = None,
                    daily_total: int | None = None,
                    focus_duration: int | None = None) -> dict:
    """把用户输入起草成对齐表。支持两种情况：
    - 完整目录粘贴 → 静态对齐（basis=user_materials）
    - 一句话（书名/博主/考试）→ 常识生成大模块（basis=general_knowledge，≤10 个知识点）
    V1.3：注入当前时间、截止日、每日总时长与专注上限，让 LLM 自己算剩余天数、
    控制总学习量（时间总铁则）。返回结构见 ALIGNMENT_SYSTEM。"""
    payload = {
        "today": _now_cn(),
        "exam_date": exam_date,
        "daily_total": daily_total,
        "focus_duration": focus_duration,
        "input": pasted_text[:4000],
    }
    raw = _chat(ALIGNMENT_SYSTEM, json.dumps(payload, ensure_ascii=False),
                max_tokens=8192)
    alignment = _as_dict(raw)
    if "resources" not in alignment or "knowledge_points" not in alignment:
        raise ValueError(f"对齐表缺少字段：{list(alignment)[:10]}")
    alignment.setdefault("basis", "user_materials")
    alignment.setdefault("subject_name", "")
    # 防御性兜底：常识生成最多 10 个大模块（MVP 铁律），超出按优先级保留
    if alignment["basis"] == "general_knowledge":
        kps = alignment["knowledge_points"]
        if len(kps) > 10:
            kps.sort(key=lambda k: -float(k.get("priority", 0.0)))
            alignment["knowledge_points"] = kps[:10]
    return alignment


def outline_note(alignment: dict) -> str:
    """常识生成大纲时的免责声明（情况 B）；静态对齐时返回空串。"""
    if alignment.get("basis") != "general_knowledge":
        return ""
    name = str(alignment.get("subject_name") or "你提到的材料").strip()
    return (f"说明：这是根据《{name}》的常规考点为你生成的初始大纲。"
            f"如果有遗漏或不匹配，你可以直接让我修改。")


ROLE_LABELS = {"learn": "听课", "practice": "刷题",
               "review": "复盘", "reference": "背诵查漏"}


def _resource_name(r) -> str:
    """资源条目归一化：字符串直接用；LLM 偶发返回对象（{name/type}）时取名称。"""
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        return str(r.get("name") or r.get("type") or "未命名资源")
    return str(r)


def render_alignment(alignment: dict) -> str:
    """把对齐表草稿渲染成用户确认用的文本；没把握的行标"?"（铁律 4）。"""
    lines = [f"资源类型：{'、'.join(_resource_name(r) for r in alignment.get('resources', []))}"]
    for kp in alignment["knowledge_points"]:
        flag = " ?" if not kp.get("confident", True) else ""
        lines.append(f"\n【{kp['name']}】权重 {kp.get('priority', 1.0)}{flag}")
        for it in kp["items"]:
            q = " ?" if not it.get("confident", True) else ""
            role = ROLE_LABELS.get(it.get("role", "learn"), "学")
            lines.append(f"  - [{role}·{it['source']}] {it['title']}（约 {it['minutes']} 分钟）{q}")
    return "\n".join(lines)


# ---------- 2. 每日一句话解析（铁律 7） ----------

PARSE_SYSTEM = PERSONA + """
你是学习计划 Agent 的汇报解析器。

用户会在晚间用一句话汇报今天的学习情况。给你今天任务清单（JSON），请把这句话解析成结构化结果，只输出一个 JSON 对象：
{
  "outcome": "done | partial | skipped",
  "partial_ratio": 0.5,
  "too_hard_kps": ["kp1"],
  "too_heavy": false,
  "misaligned": false,
  "note": "一句温柔回应"
}

规则：
1. outcome："做完了/完成了"→done；"做了一半/只看了/部分"→partial（partial_ratio 给完成比例估计 0-1）；"没学/没做/没看/废了"→skipped。
2. "太难/看不懂/没看懂/不会"→ 把对应知识点 id 放进 too_hard_kps（只能从上下文的 kp_id 里选；无法确定时填 "__today__"；没有就填空数组）。
3. "太重/做不完/任务多"→ too_heavy=true（此时 outcome 通常是 skipped 或 partial，按语气判断）。
4. "对不上/视频里没讲/题和视频不匹配"→ misaligned=true。
5. 保持克制：用户没说的事不要脑补。
6. note：一句口语化的温柔回应（如『辛苦啦，剩下的交给我』『没关系，明天轻一点』），不超过 20 字，绝不催债、绝不说教。

只输出 JSON，不要输出任何其他文字。"""


def parse_report(text: str, blocks: list | None = None) -> dict:
    """每日一句话解析。输出结构与 planner.parse_report 完全一致：
    {outcome, partial_ratio?, too_hard_kps, too_heavy, misaligned, _text}，
    可被 planner.apply_report 原样消费。

    blocks（可选）：今天任务块的上下文，让 LLM 把"太难"精确映射到 kp_id；
    不传时行为与占位版相同（too_hard_kps 填 "__today__"）。
    LLM 失败时自动回退到 planner 的关键词占位实现，晚间循环不中断。
    """
    from planner import parse_report as keyword_parse_report  # 延迟导入，避免循环依赖
    try:
        ctx = {"report": text, "today_blocks": blocks or []}
        raw = _chat(PARSE_SYSTEM, json.dumps(ctx, ensure_ascii=False), max_tokens=1024)
        result = _as_dict(raw)
        return {
            "outcome": result.get("outcome", "done"),
            "partial_ratio": float(result.get("partial_ratio", 0.5)),
            "too_hard_kps": list(result.get("too_hard_kps") or []),
            "too_heavy": bool(result.get("too_heavy", False)),
            "misaligned": bool(result.get("misaligned", False)),
            "note": str(result.get("note", "") or "").strip(),
            "_text": text,
        }
    except Exception:
        return keyword_parse_report(text)


# ---------- 3. 预警与砍内容文案（铁律 10） ----------

WARNING_SYSTEM = PERSONA + """
你是学习计划 Agent 的预警撰稿人。

计划引擎算出：按当前进度，截止日前学不完了。请把预警写成两到三句话 + 建议清单，语气像一位冷静温柔、懂心理学的教练。

铁律：
1. 如实但不说教、不制造负罪感；禁用"逾期"等催债措辞。
2. 说清事实：还剩几天、还差多少分钟排不下。
3. 砍内容建议必须说明理由（结合知识点名字给出合理解释；解释要诚实，不要编造统计数据）。
4. 结尾必须给"不砍"选项：用户说"不砍"就维持全量计划。
5. 全篇不超过 120 字。只输出最终文案，不要输出其他文字。"""


def write_warning_copy(warning: dict) -> str:
    """预警 + 砍内容建议的最终文案（铁律 10）。LLM 失败时用确定性文案兜底。"""
    payload = {
        "days_left": warning["days_left"],
        "shortfall_min": warning["shortfall_min"],
        "proposed_cuts": [
            {"kp_name": c["kp_name"], "minutes_saved": c["minutes_saved"]}
            for c in warning.get("proposed_cuts", [])
        ],
    }
    try:
        return _chat(
            WARNING_SYSTEM, json.dumps(payload, ensure_ascii=False), max_tokens=1024
        ).strip()
    except Exception:
        cuts = "、".join(c["kp_name"] for c in warning.get("proposed_cuts", []))
        return (
            f"如实说：还剩 {warning['days_left']} 天，按当前速度约 {warning['shortfall_min']} 分钟排不下。"
            f"建议砍掉 {cuts}（可释放时间），也可以回『不砍』维持全量计划。"
        )


# ---------- 4. （V1.2 起启动确认并入 extract()，finalize_setup 已废弃删除） ----------

_load_dotenv()
