"""evening_reminder.py —— 每天晚 8 点的晚间汇报提醒（方案 B：独立 CLI + 外部定时器）。

设计要点（与 app.py 解耦，不复用 uvicorn 生命周期）：
1. 遍历 SQLite agent.db 的 user_states 表，为每个用户生成一条晚间提醒；
2. 提醒同时写两处（用户拍板②）：
   - state["chat"] 追加一条 agent 消息（前端打开即见）；
   - state["history"] 追加一条 {"type": "reminder"} 记录（持久留存）；
3. 幂等：last_reminded_at 存在 user_states.last_reminded_at 列（用户拍板③），
   当天已提醒过的会话直接跳过，重跑/补跑不会重复打扰；
4. 无活跃计划（phase=new 或没有 deadline/排期）的用户发轻量促活文案（拍板①）；
   已有计划的用户发个性化提醒：进度看板 + 今晚汇报引导。
5. 纯确定性文案，不调 LLM（晚间循环绝不因 API 故障而中断）。

用法：
  python evening_reminder.py               # 正式运行（写库）
  python evening_reminder.py --dry-run     # 只打印将写入的内容，不写库
  python evening_reminder.py --db PATH     # 指定数据库文件（测试用副本）
  python evening_reminder.py --date YYYY-MM-DD  # 手动补跑指定日期

退出码：0 = 正常；1 = 有异常。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import database
from state import validate
from planner import daily_blocks, progress_board, render_blocks

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROMO_TEXT = (
    "嗨，教练还在等你哦～准备好材料（书名、目录或网课标题都行）随时发我，"
    "我们就可以开始搭大纲排计划啦🌙"
)


def _today_blocks_text(s: dict, day: str) -> str:
    blocks = daily_blocks(s, day)
    if not blocks:
        return "（今天没有排任务）"
    return "\n".join(render_blocks(blocks))


def _build_message(s: dict, day: str) -> str:
    """按用户状态生成提醒文案（确定性，无 LLM）。"""
    flow = s.get("flow", {})
    phase = flow.get("phase", "new")
    has_plan = bool(s["meta"].get("deadline")) and bool(s.get("schedule"))

    if not has_plan or phase == "new":
        return PROMO_TEXT

    board = progress_board(s, day)
    parts = ["晚上好～今晚用一句话汇报一下今天就行（比如『做了一半』『太难了』『没学』都行）。",
             "今天的清单：\n" + _today_blocks_text(s, day)]
    if board:
        parts.append(board)
    reported = any(e.get("type") == "report" and e.get("date") == day
                   for e in s.get("history", []))
    if reported:
        parts.append("今天已经汇报过啦，辛苦了，安心休息～")
    if flow.get("phase") == "await_cut":
        parts.append("上次的预警还等你拍板：回『砍』按建议来，或『不砍』维持全量，都听你的。")
    return "\n\n".join(parts)


def _is_remindable(s: dict) -> bool:
    """结构完整、能校验通过的状态才提醒；损坏数据跳过并记录。"""
    try:
        validate(s)
        s.setdefault("chat", [])
        s.setdefault("history", [])
        return True
    except (KeyError, ValueError):
        return False


def run(db_path: str | None = None, dry_run: bool = False,
        day: str | None = None) -> int:
    if db_path:
        database.set_db_path(db_path)
    database.init_db()

    today = day or date.today().isoformat()
    sessions = database.iter_sessions()
    print(f"晚间提醒 · {today} · 共 {len(sessions)} 个会话"
          + ("（dry-run，不写库）" if dry_run else ""))
    print("=" * 64)

    reminded = skipped_idem = skipped_bad = 0
    errors = []
    for sid, raw in sessions:
        try:
            s = json.loads(raw)
        except json.JSONDecodeError as e:
            skipped_bad += 1
            errors.append(f"{sid}: state_json 解析失败 {e}")
            continue
        if not _is_remindable(s):
            skipped_bad += 1
            errors.append(f"{sid}: state 结构校验未通过（跳过）")
            continue
        if database.get_last_reminded(sid) == today:
            skipped_idem += 1
            print(f"[跳过] {sid}（{today} 已提醒过）")
            continue

        msg = _build_message(s, today)
        now = datetime.now().isoformat(timespec="seconds")
        chat_msg = {"role": "agent", "text": msg, "time": today}
        history_entry = {"date": today, "type": "reminder",
                         "text": msg, "detail": {}, "_at": now}

        print(f"[{'dry-run' if dry_run else '提醒'}] {sid}：")
        for line in msg.splitlines():
            print("    " + line)

        if not dry_run:
            s["chat"].append(chat_msg)
            s["history"].append(history_entry)
            validate(s)
            database.save_state(sid, json.dumps(s, ensure_ascii=False))
            database.set_last_reminded(sid, today)
        reminded += 1

    print("=" * 64)
    print(f"结果：提醒 {reminded}，幂等跳过 {skipped_idem}，"
          f"坏数据跳过 {skipped_bad}，异常 {len(errors)}")
    for e in errors:
        print("  - " + e)
    return 0


def main():
    ap = argparse.ArgumentParser(description="晚间汇报提醒（幂等，可重跑）")
    ap.add_argument("--db", help="数据库文件路径（默认项目根 agent.db）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写库")
    ap.add_argument("--date", help="手动指定提醒日期 YYYY-MM-DD（补跑用）")
    args = ap.parse_args()
    try:
        sys.exit(run(db_path=args.db, dry_run=args.dry_run, day=args.date))
    except Exception as e:
        print(f"[FAIL] 运行失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
