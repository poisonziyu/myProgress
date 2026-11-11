"""smoke_test.py —— 离线冒烟测试（开发铁序第一步）。

用假数据在终端跑通四件事：
  场景 1  生成任务：首日产出 3-4 个任务块，带预计时长（铁律 6）
  场景 2  一句话汇报 → 自动顺延（铁律 7、8）
  场景 3  抱怨太重 → 容量画像下调（铁律 9）
  场景 4  连续落后 → 提前预警 + 砍内容（铁律 10、11）

运行：python smoke_test.py
退出码：0 = 全部通过；1 = 有失败（并逐条列出）。
"""

import sys
import tempfile

import database
from state import default_state, save_state, load_state
from planner import (
    apply_cut_decision, apply_report, check_warning, daily_blocks,
    init_plan, parse_report, render_blocks,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS = 0
FAILURES = []


def ok(cond, label):
    global CHECKS
    CHECKS += 1
    mark = "[OK]  " if cond else "[FAIL]"
    print(f"    {mark} {label}")
    if not cond:
        FAILURES.append(label)


def make_kp(kp_id, name, priority, minutes=(25, 20, 10)):
    """造一个知识点：学（视频）→ 练（题库）→ 查漏（教材）三个条目。
    资源类型名是用户自定义的（铁律 5），这里只是假数据示例。"""
    v, p, r = minutes
    return {
        "id": kp_id,
        "name": name,
        "priority": priority,   # 性价比权重：越大越不能砍
        "status": "pending",
        "items": [
            {"id": f"{kp_id}-v", "source": "视频源", "role": "learn",
             "title": f"{name} 讲解视频 P1", "minutes": v, "remaining_min": v, "status": "todo"},
            {"id": f"{kp_id}-p", "source": "习题库", "role": "practice",
             "title": f"{name} 练习题 15 道", "minutes": p, "remaining_min": p, "status": "todo"},
            {"id": f"{kp_id}-r", "source": "主教材", "role": "reference",
             "title": f"{name} 教材对应章节", "minutes": r, "remaining_min": r, "status": "todo"},
        ],
    }


def scenario1():
    print("\n" + "=" * 64)
    print("场景 1：生成任务（铁律 6：3-4 个任务块，带预计时长）")
    state = default_state(deadline="2026-12-20", life_profile="打工人", daily_total=120, start_date="2026-11-05")
    state["materials"]["resources"] = ["视频源", "主教材", "习题库"]
    state["knowledge_points"] = [
        make_kp("kp1", "资料分析-增长率", 1.0),
        make_kp("kp2", "资料分析-比重", 0.9),
        make_kp("kp3", "资料分析-平均数", 0.8),
        make_kp("kp4", "数量关系-工程问题", 0.3),
    ]
    result = init_plan(state)
    day1 = daily_blocks(state, result["first_day"])
    ok(len(day1) == 4, f"首日任务块数 = 4（实际 {len(day1)}）")
    ok(result["first_day"] == "2026-11-05", "确认当天就是首日（今日清单，非明日）")
    ok(all(b["minutes"] > 0 for b in day1), "每个任务块都带预计时长")
    ok(result["warning"] is None, "进度赶得上时绝不触发预警")
    print("    首日清单（2026-11-05，确认当天）：")
    for line in render_blocks(day1):
        print("      " + line)

    # V1.4：SQLite 读写回环（临时库，不碰真实 agent.db）
    with tempfile.TemporaryDirectory() as td:
        database.set_db_path(f"{td}/test.db")
        save_state(state, "sess-roundtrip")
        loaded = load_state("sess-roundtrip")
    ok(len(loaded["knowledge_points"]) == 4 and loaded["schedule"],
       "SQLite（agent.db 同构临时库）保存/读取回环正常")


def scenario2():
    print("\n" + "=" * 64)
    print("场景 2：晚间一句话汇报 → 自动顺延（铁律 7、8）")
    state = default_state(deadline="2026-12-20", life_profile="打工人", daily_total=120, start_date="2026-11-05")
    state["knowledge_points"] = [
        make_kp("kp1", "资料分析-增长率", 1.0),
        make_kp("kp2", "资料分析-比重", 0.9),
    ]
    init_plan(state)
    day1 = daily_blocks(state, "2026-11-05")

    parsed = parse_report("今天只看了一半")
    ok(parsed["outcome"] == "partial", f"解析结果 = partial（实际 {parsed['outcome']}）")
    summary = apply_report(state, "2026-11-05", parsed)
    for line in summary["summary"]:
        print("    Agent:", line)

    ok(len(day1) == 4 and all(b["status"] == "partial" for b in day1),
       "当天 4 个任务块全部标记 partial，历史日程未被抹掉")
    day2 = daily_blocks(state, "2026-11-06")
    kp1_video = next((b for b in day2 if b["item_id"] == "kp1-v"), None)
    ok(kp1_video is not None, "未完成的视频任务顺延到了明天")
    ok(kp1_video is not None and 0 < kp1_video["minutes"] < 25,
       f"顺延后显示的是剩余量（约 {kp1_video['minutes'] if kp1_video else '?'} 分钟，< 25）")
    ok(not any("逾期" in line for line in render_blocks(day2)), "文案不含『逾期』（铁律 8）")
    print("    顺延后的明日清单（2026-11-06）：")
    for line in render_blocks(day2):
        print("      " + line)


def scenario3():
    print("\n" + "=" * 64)
    print("场景 3：抱怨『太重了』→ 容量画像下调（铁律 9）")
    state = default_state(deadline="2026-12-20", life_profile="打工人", daily_total=120, start_date="2026-11-09")
    state["knowledge_points"] = [
        make_kp("kp1", "资料分析-增长率", 1.0),
        make_kp("kp2", "资料分析-比重", 0.9),
    ]
    init_plan(state)
    before = state["config"]["daily_total"]

    parsed = parse_report("今天的任务太重了，做不完")
    ok(parsed["too_heavy"], "解析出 too_heavy 信号")
    summary = apply_report(state, "2026-11-10", parsed)
    for line in summary["summary"]:
        print("    Agent:", line)

    after = state["config"]["daily_total"]
    ok(after == 84, f"每日总时长 120 → 84（×0.7，实际 {before} → {after}）")
    day2 = daily_blocks(state, "2026-11-11")
    total = sum(b["minutes"] for b in day2)
    ok(total <= after, f"明日总任务量 {total} 分钟 ≤ 新容量 {after} 分钟")
    print("    调整后的明日清单（2026-11-11）：")
    for line in render_blocks(day2):
        print("      " + line)


def scenario4():
    print("\n" + "=" * 64)
    print("场景 4：连续落后 → 提前预警 + 砍内容（铁律 10、11）")
    state = default_state(deadline="2026-11-09", life_profile="打工人", daily_total=120, start_date="2026-11-05")
    state["knowledge_points"] = [
        make_kp("kp1", "资料分析-增长率", 1.0, minutes=(30, 30, 20)),
        make_kp("kp2", "资料分析-比重", 0.9, minutes=(30, 30, 20)),
        make_kp("kp3", "数量关系-工程问题", 0.8, minutes=(30, 30, 20)),
    ]
    init_plan(state)
    ok(not state["warnings"], "初始进度赶得上，无预警")

    apply_report(state, "2026-11-06", parse_report("今天没学，加班废了"))
    summary = apply_report(state, "2026-11-07", parse_report("还是没学，状态很差"))
    for line in summary["summary"]:
        print("    Agent:", line)

    leftover = summary["unscheduled"]
    ok(bool(leftover), "连续落后两天后，出现了排不下的内容")
    warning = check_warning(state, "2026-11-07", leftover)
    ok(warning is not None, "『考前学不完』在数学上成立 → 立即预警")
    if warning:
        ok(warning["status"] == "pending", "预警进入待决策状态")
        ok(warning["option_keep_all"], "始终给『不砍』选项（铁律 10）")
        ok(len(warning["proposed_cuts"]) == 2,
           f"贪心找到最小砍集合（实际 {len(warning['proposed_cuts'])} 个）")
        ok(all("性价比" in c["reason"] for c in warning["proposed_cuts"]),
           "每条建议都附理由")
        ok(not any("逾期" in c["reason"] for c in warning["proposed_cuts"]),
           "预警文案不含『逾期』（铁律 8）")
        print(f"    Agent 的预警（还剩 {warning['days_left']} 天，排不下约 {warning['shortfall_min']} 分钟）：")
        for c in warning["proposed_cuts"]:
            print(f"      建议砍掉：{c['reason']}")
        print("      也可以回『不砍』，维持全量计划。")

        result = apply_cut_decision(state, "2026-11-07", "accept")
        ok(result["status"] == "accepted", "用户确认后预警状态 = accepted")
        cut_ids = {c["kp_id"] for c in warning["proposed_cuts"]}
        for kp in state["knowledge_points"]:
            if kp["id"] in cut_ids:
                ok(kp["status"] == "cut", f"{kp['name']} 已被砍掉")
        ok(not result["unscheduled"], "砍完后全部内容都能在考前排下")
        day3 = daily_blocks(state, "2026-11-08")
        print("    砍完重排后的考前最后一天（2026-11-08）：")
        for line in render_blocks(day3):
            print("      " + line)


def main():
    print("myProgress 离线冒烟测试")
    scenario1()
    scenario2()
    scenario3()
    scenario4()
    print("\n" + "=" * 64)
    if FAILURES:
        print(f"结果：{len(FAILURES)}/{CHECKS} 项未通过")
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print(f"结果：全部通过（{CHECKS} 项检查）")


if __name__ == "__main__":
    main()
