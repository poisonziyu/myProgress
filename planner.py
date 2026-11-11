"""planner.py —— 确定性计划引擎（铁律：排程是纯 Python，可离线测试）。

V1.3（通用学习教练）：
1. 时间总铁则：每天的总耗时 = 纯学习时长 + 休息时长（每块后固定 break_minutes），
   生成清单时自动核算全部耗时，超上限就删减任务，严禁堆积。
2. 专注上限：单块学习时长 ≤ config.focus_duration（默认 25 分钟番茄），超过自动拆分。
3. 学习闭环：同一天优先同一个模块闭环（输入→练习→复盘），再跨模块交替，降低认知消耗。
4. 长期阶段自动划分：基础筑基期 → 强化练习期 → 综合冲刺期（按剩余天数比例），
   阶段决定角色排序（筑基学在前、强化练在前、冲刺复盘在前）。
5. 顺延：未完成块 + 待排条目合并成任务池，从明天起重排，绝不催债。
6. 预警：截止日前学不完在数学上成立时立即预警 + 砍内容建议 + 永远给"不砍"选项。
"""

from __future__ import annotations

import copy
from datetime import date, timedelta

WEEKEND_MULTIPLIER = 1.5   # 上班族周末可多投入（config.weekend_boost 时生效）
MIN_TOTAL_MINUTES = 30     # 总时长下限："太重了"不能把计划压到归零
MAX_BLOCKS_PER_DAY = 4     # 每日 3-4 个任务块（软上限；造成"假性学不完"时自动放开）

# V1.3 四类动作（统一任务输出格式自动区分）
ROLE_LABELS = {"learn": "听课学习", "practice": "刷题实操",
               "review": "复盘整理", "reference": "背诵查漏"}

SCHEDULABLE_STATUSES = ("pending", "stuck", "in_progress")

PHASE_NAMES = {1: "基础筑基期", 2: "强化练习期", 3: "综合冲刺期"}

REVIEW_GUIDE = "复盘自查：①概念没理解？②步骤不熟练？③粗心失误？④思路不会？"

CUT_REASON_TEMPLATE = (
    "『{name}』性价比相对最低（权重 {priority}），砍掉可释放约 {minutes} 分钟"
)


# ---------- 小工具 ----------

def _as_date(d) -> date:
    return d if isinstance(d, date) else date.fromisoformat(d)


def _log(state, day, type_, text, detail=None):
    state["history"].append(
        {"date": day, "type": type_, "text": text, "detail": detail or {}}
    )


def _find_kp(state, kp_id):
    return next((k for k in state["knowledge_points"] if k["id"] == kp_id), None)


def _find_item(state, item_id):
    for kp in state["knowledge_points"]:
        for it in kp["items"]:
            if it["id"] == item_id:
                return it
    return None


def current_phase(state, d=None) -> int:
    """长期阶段：按已过时间占全程的比例划分（通用，不分学科）。
    1=基础筑基期（<50%）、2=强化练习期（50%-85%）、3=综合冲刺期（≥85%）。"""
    start = _as_date(state["meta"]["start_date"])
    deadline = _as_date(state["meta"]["deadline"])
    d = _as_date(d) if d else date.today()
    total = max(1, (deadline - start).days)
    ratio = (d - start).days / total
    return 1 if ratio < 0.5 else (2 if ratio < 0.85 else 3)


def progress_board(state, d=None) -> str:
    """精简进度看板：距截止天数｜当前阶段｜模块完成情况。"""
    if not state["meta"].get("deadline"):
        return ""
    d = _as_date(d) if d else date.today()
    deadline = _as_date(state["meta"]["deadline"])
    days_left = (deadline - d).days
    done = [k["name"] for k in state["knowledge_points"] if k["status"] == "done"]
    total = len(state["knowledge_points"])
    parts = [f"距截止 {days_left} 天", f"当前阶段：{PHASE_NAMES[current_phase(state, d)]}",
             f"模块完成 {len(done)}/{total}"]
    if done:
        parts.append("已完成：" + "、".join(f"『{n}』" for n in done))
    return "📋 " + "｜".join(parts)


def _role_order(state, d) -> dict:
    """各阶段的动作优先级：筑基=学在前、强化=练在前、冲刺=复盘在前。"""
    phase = current_phase(state, d)
    if phase == 1:
        return {"learn": 0, "practice": 1, "reference": 2, "review": 3}
    if phase == 2:
        return {"practice": 0, "review": 1, "learn": 2, "reference": 3}
    return {"review": 0, "practice": 1, "reference": 2, "learn": 3}


def day_capacity(state, d) -> int:
    """某天的总时长上限（分钟，含休息）= config.daily_total；上班族周末 ×1.5。"""
    d = _as_date(d)
    cfg = state["config"]
    base = int(cfg["daily_total"])
    if cfg.get("weekend_boost") and d.weekday() >= 5:
        return round(base * WEEKEND_MULTIPLIER)
    return base


def _pending_pairs(state, d=None):
    """待排条目：可调度知识点 × 未完成条目，按性价比降序、阶段角色顺序排列。"""
    kps = [k for k in state["knowledge_points"] if k["status"] in SCHEDULABLE_STATUSES]
    kps.sort(key=lambda k: -k["priority"])
    order = _role_order(state, d or date.today())
    pairs = []
    for kp in kps:
        items = [it for it in kp["items"]
                 if it["status"] != "done" and it["remaining_min"] > 0]
        items.sort(key=lambda it: order.get(it["role"], 9))
        pairs.extend((kp, it) for it in items)
    return pairs


def _chunk_queue(state, pairs):
    """把待排条目切成 ≤ focus_duration 的番茄小块（专注上限，超过必须拆分）。
    返回 [(kp, item, chunk_minutes), ...]，同一 item 可占多块。"""
    focus = max(10, int(state["config"]["focus_duration"]))
    queue = []
    for kp, it in pairs:
        rem = it["remaining_min"]
        while rem > 0:
            chunk = min(rem, focus)
            queue.append((kp, it, chunk))
            rem -= chunk
    return queue


def _pick_index(queue, used_cost, cap_total, break_min, last_kp_id, last_role):
    """选下一块（V1.3 闭环优先）：
    ① 同模块不同角色（当天闭环：输入→练习→复盘，少跨模块）
    ② 不同模块不同角色（换脑子）
    ③ 队首。
    每块成本 = 纯学习 + 固定休息（时间总铁则）。"""
    def fits(chunk):
        return used_cost + chunk + break_min <= cap_total

    if last_kp_id is not None:
        for i in range(min(6, len(queue))):
            kp, it, c = queue[i]
            if kp["id"] == last_kp_id and it["role"] != last_role and fits(c):
                return i
    for i in range(min(3, len(queue))):
        kp, it, c = queue[i]
        if (last_kp_id is None or kp["id"] != last_kp_id) \
                and it["role"] != last_role and fits(c):
            return i
    if queue and fits(queue[0][2]):
        return 0
    return -1


def _fill_day(queue, cap_total, cap_blocks, break_min):
    """从队列取出装得进今天的块（原地消耗 queue）。
    成本 = 纯学习 + 每块后固定休息；返回 [(kp, item, chunk), ...]。"""
    blocks = []
    used_cost = 0
    last_kp_id, last_role = None, None
    while queue:
        if cap_blocks and len(blocks) >= MAX_BLOCKS_PER_DAY:
            break
        i = _pick_index(queue, used_cost, cap_total, break_min,
                        last_kp_id, last_role)
        if i < 0:
            break
        kp, it, chunk = queue.pop(i)
        blocks.append((kp, it, chunk))
        used_cost += chunk + break_min
        last_kp_id, last_role = kp["id"], it["role"]
    return blocks


def _make_block(kp, it, chunk, n) -> dict:
    return {
        "id": f"{it['id']}#{n}",
        "kp_id": kp["id"],
        "kp_name": kp["name"],
        "item_id": it["id"],
        "source": it["source"],
        "role": it["role"],
        "title": it["title"],
        "minutes": chunk,          # 纯学习时长（已按 focus_duration 切块）
        "status": "todo",
    }


def _plan_from_pairs(state, pairs, from_d):
    """纯计算：从 from_d 排到截止日前一天。返回 (日程, 未排入的块队列)。
    先按 3-4 块软上限排；若因此出现"假性学不完"，放开上限重排一次。"""
    deadline = _as_date(state["meta"]["deadline"])
    break_min = int(state["config"].get("break_minutes", 5))
    queue = _chunk_queue(state, list(pairs))
    schedule = {}
    d = from_d
    while d < deadline and queue:
        blocks = _fill_day(queue, day_capacity(state, d), cap_blocks=True, break_min=break_min)
        if blocks:
            schedule[d.isoformat()] = {
                "blocks": [_make_block(kp, it, ch, n)
                           for n, (kp, it, ch) in enumerate(blocks, start=1)]
            }
        d += timedelta(days=1)
    if queue:
        queue2 = _chunk_queue(state, _pending_pairs(state, from_d))
        schedule2 = {}
        d = from_d
        while d < deadline and queue2:
            blocks = _fill_day(queue2, day_capacity(state, d), cap_blocks=False, break_min=break_min)
            if blocks:
                schedule2[d.isoformat()] = {
                    "blocks": [_make_block(kp, it, ch, n)
                               for n, (kp, it, ch) in enumerate(blocks, start=1)]
                }
            d += timedelta(days=1)
        if len(queue2) < len(queue):
            return schedule2, queue2
    return schedule, queue


# ---------- 对外接口 ----------

def init_plan(state, from_date=None) -> dict:
    """启动排程：从确认当天排起（确认后拿到的必须是【今日清单】；
    晚间汇报后才生成【明日清单】）。
    from_date 缺省用 meta.start_date（离线测试可注入固定日期）。"""
    first_day = from_date or state["meta"]["start_date"]
    leftover = reschedule(state, first_day)
    warning = check_warning(state, first_day, leftover)
    return {"first_day": first_day, "unscheduled": leftover, "warning": warning}


def reschedule(state, from_date):
    """从 from_date 起重排；早于该日的日程保留（顺延不改历史）。
    返回未排入的块队列（非空 = 截止日前排不下）。"""
    from_d = _as_date(from_date)
    schedule, leftover = _plan_from_pairs(state, _pending_pairs(state, from_d), from_d)
    for key in [k for k in list(state["schedule"]) if _as_date(k) >= from_d]:
        del state["schedule"][key]
    state["schedule"].update(schedule)
    return leftover


def daily_blocks(state, date_str) -> list:
    return state["schedule"].get(date_str, {}).get("blocks", [])


def render_blocks(blocks) -> list[str]:
    """V1.3 统一任务输出格式：
    □ 任务名称（纯学习时长）
    └ 休息：5-10 分钟
    复盘任务附带复盘自查引导（禁止只写"复盘"两个字）。"""
    lines = []
    for b in blocks:
        role = ROLE_LABELS.get(b["role"], b["role"])
        lines.append(f"□ {role}：{b['title']}（纯学习 {b['minutes']} 分钟）")
        if b["role"] == "review":
            lines.append(f"　└ {REVIEW_GUIDE}")
        lines.append("　└ 休息：5-10 分钟")
    return lines


def parse_report(text: str) -> dict:
    """离线版一句话解析（占位 + LLM 挂掉时的兜底）。
    主路径是 llm.extract 的 report 字段，返回结构不变。"""
    parsed = {
        "outcome": "done",
        "too_hard_kps": [],
        "too_heavy": False,
        "misaligned": False,
        "_text": text,
    }
    if any(w in text for w in ("对不上", "没讲")):
        parsed["misaligned"] = True
    if any(w in text for w in ("太难", "看不懂", "没看懂", "不会", "卡住")):
        parsed["too_hard_kps"] = ["__today__"]
    if any(w in text for w in ("太重", "任务重", "做不完", "完不成", "扛不住", "吃不消")):
        parsed["too_heavy"] = True
        parsed["outcome"] = "skipped"
    elif any(w in text for w in ("没看完", "没做完", "没完成", "没弄完", "只看了一半", "只做了一半")):
        parsed["outcome"] = "partial"
    elif any(w in text for w in ("一半", "部分", "一截")):
        parsed["outcome"] = "partial"
    elif any(w in text for w in ("没学", "没看", "没做", "没动", "啥也没", "废了", "状态不好")):
        parsed["outcome"] = "skipped"
    return parsed


def _join_names(titles: list) -> str:
    """把『A』『B』『C』拼成口语化的『A』『B』 和 『C』。"""
    if not titles:
        return ""
    if len(titles) == 1:
        return titles[0]
    return "、".join(titles[:-1]) + " 和 " + titles[-1]


def apply_report(state, today, parsed) -> dict:
    """执行晚间汇报（块级合并版）：

    1. 逐块标记：outcome=done 或 done_blocks 里点名的块 → done；
       其余 partial 块 → partial，skipped → skipped。
    2. 按块扣减条目剩余量（done 全扣、partial 按比例扣）——"前两部分完成了"
       只扣前两块，剩下的完整顺延，绝不让清单假顺延。
    3. 未完成块 + 全部待排条目合并成一个任务池，从明天起重排（reschedule）。
    4. 回复中明确列出顺延合并了什么（让用户有真实感知）。

    返回 {"summary": [...], "unscheduled": [...]}。"""
    today_blocks = state["schedule"].get(today, {}).get("blocks", [])
    summary = []
    outcome = parsed["outcome"]
    done_indexes = set()
    for i in parsed.get("done_blocks") or []:
        try:
            done_indexes.add(int(i))
        except (TypeError, ValueError):
            pass
    ratio = max(0.0, min(1.0, float(parsed.get("partial_ratio", 0.5))))

    # 1. 逐块标记（1 起始序号与提取器的约定一致）
    for idx, blk in enumerate(today_blocks, start=1):
        if blk["status"] == "done":
            continue
        if outcome == "done" or idx in done_indexes:
            blk["status"] = "done"
        elif outcome == "partial":
            blk["status"] = "partial"
        else:
            blk["status"] = "skipped"

    # 2. 按块扣减条目剩余量
    for blk in today_blocks:
        if blk["status"] not in ("done", "partial"):
            continue
        it = _find_item(state, blk["item_id"])
        if it is None or it["remaining_min"] <= 0:
            continue
        done_min = blk["minutes"] if blk["status"] == "done" else round(blk["minutes"] * ratio)
        it["remaining_min"] = max(0, it["remaining_min"] - done_min)
        if it["remaining_min"] == 0:
            it["status"] = "done"

    # 3. 顺延合并的真实感知：明确列出没做完的块
    carried = []
    for blk in today_blocks:
        if blk["status"] == "done":
            continue
        it = _find_item(state, blk["item_id"])
        if it is None:
            continue
        t = f"『{blk['title']}』"
        if t not in carried:
            carried.append(t)

    if outcome == "done":
        summary.append("今天全部完成，干得漂亮！")
    elif carried:
        summary.append(
            f"我把你今天没做完的 {_join_names(carried)}，顺延并合并到了明天。"
        )
    elif outcome == "partial":
        summary.append("没做完的部分我自动顺延好了，不用有负担～")
    else:
        summary.append("今天的任务顺延到后面了，好好休息，明天继续～")

    for kp_id in parsed.get("too_hard_kps", []):
        if kp_id == "__today__":
            blk = next((b for b in today_blocks if b["status"] != "done"), None)
            kp_id = blk["kp_id"] if blk else None
        kp = _find_kp(state, kp_id) if kp_id else None
        if kp and kp["status"] != "cut":
            kp["status"] = "stuck"
            summary.append(f"『{kp['name']}』先标记为卡点，我会放慢拆分处理，不着急。")

    if parsed.get("too_heavy"):
        changed = adjust_capacity(
            state, 0.7, f"用户反馈任务太重：{parsed['_text']}", today
        )
        if changed:
            summary.append(
                f"最近负担有点重，我把每天的总时长调小了些：{changed[0]} → {changed[1]} 分钟。"
            )

    if parsed.get("misaligned"):
        _log(state, today, "misaligned", "用户反馈材料对不上，等待对齐映射修正")

    _refresh_kp_statuses(state)

    tomorrow = (_as_date(today) + timedelta(days=1)).isoformat()
    leftover = reschedule(state, tomorrow)

    _log(state, today, "report", parsed["_text"], {
        "outcome": outcome,
        "too_heavy": parsed["too_heavy"],
        "too_hard": parsed["too_hard_kps"],
    })
    return {"summary": summary, "unscheduled": leftover}


def adjust_capacity(state, factor, reason, today=None):
    """调整每日总时长（config.daily_total），返回 (旧, 新) 或 None（已达下限）。"""
    cur = int(state["config"]["daily_total"])
    new = max(MIN_TOTAL_MINUTES, round(cur * factor))
    if new == cur:
        return None
    state["config"]["daily_total"] = new
    state["config"]["history"].append({
        "date": today or date.today().isoformat(),
        "reason": reason,
        "daily_total": new,
    })
    return (cur, new)


def check_warning(state, today, leftover):
    """预警：当"截止日前学不完"在数学上成立时，立即预警。
    leftover = reschedule() 的未排入块队列 [(kp, item, chunk), ...]。
    返回预警 dict 或 None。"""
    if not leftover:
        return None
    today_d = _as_date(today)
    deadline = _as_date(state["meta"]["deadline"])
    days_left = max(0, (deadline - today_d).days - 1)
    shortfall = sum(ch for _, _, ch in leftover)
    warning = {
        "date": today,
        "days_left": days_left,
        "shortfall_min": shortfall,
        "leftover": [
            {"kp": kp["name"], "item": it["title"], "min": ch}
            for kp, it, ch in leftover
        ],
        "proposed_cuts": _propose_cuts(state, today_d),
        "option_keep_all": True,   # 永远给"不砍"选项
        "status": "pending",
    }
    state["warnings"].append(warning)
    return warning


def _propose_cuts(state, today_d):
    """贪心找最小砍内容集合：从性价比最低的知识点开始模拟砍，直到能排下。
    reason 为占位模板，预警文案由 llm.write_warning_copy 写真实理由。"""
    from_d = today_d + timedelta(days=1)
    sim = copy.deepcopy(state)
    candidates = [k for k in sim["knowledge_points"] if k["status"] in SCHEDULABLE_STATUSES]
    candidates.sort(key=lambda k: k["priority"])   # 性价比最低的先试砍
    cuts = []
    for kp in candidates:
        saved = sum(it["remaining_min"] for it in kp["items"] if it["status"] != "done")
        kp["status"] = "cut"                        # 在模拟副本上累积砍
        cuts.append({
            "kp_id": kp["id"],
            "kp_name": kp["name"],
            "minutes_saved": saved,
            "reason": CUT_REASON_TEMPLATE.format(
                name=kp["name"], priority=kp["priority"], minutes=saved
            ),
        })
        _, left2 = _plan_from_pairs(sim, _pending_pairs(sim, from_d), from_d)
        if not left2:
            break
    return cuts


def apply_cut_decision(state, today, decision="accept") -> dict:
    """用户对预警的回应：accept 按提案砍；reject 全保留（选择权在用户）。
    返回 {"status": ..., "unscheduled": [...]}。"""
    warning = None
    for w in reversed(state["warnings"]):
        if w["date"] == today and w["status"] == "pending":
            warning = w
            break
    if warning is None:
        raise ValueError(f"{today} 没有待处理的预警")
    if decision == "accept":
        for c in warning["proposed_cuts"]:
            kp = _find_kp(state, c["kp_id"])
            if kp:
                kp["status"] = "cut"
        warning["status"] = "accepted"
        names = "、".join(c["kp_name"] for c in warning["proposed_cuts"])
        _log(state, today, "cut", f"用户确认砍掉：{names}")
    else:
        warning["status"] = "rejected"
        _log(state, today, "cut", "用户选择不砍，维持全量计划")
    tomorrow = (_as_date(today) + timedelta(days=1)).isoformat()
    leftover = reschedule(state, tomorrow)
    return {"status": warning["status"], "unscheduled": leftover}


def _refresh_kp_statuses(state):
    for kp in state["knowledge_points"]:
        if kp["status"] == "cut":
            continue
        items = kp["items"]
        if all(it["status"] == "done" for it in items):
            kp["status"] = "done"
        elif any(it["status"] in ("done", "partial") for it in items):
            if kp["status"] != "stuck":
                kp["status"] = "in_progress"
