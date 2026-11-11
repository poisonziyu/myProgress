"""llm_smoke_test.py —— llm.py 的冒烟测试。

Part A（离线，不花 API 钱）：验证"对齐表 → 状态 → 排程"结构链路，以及
  LLM 形状的解析结果能被 planner.apply_report 原样消费（接口纪律）。
Part B（联机，需要 LLM_API_KEY + LLM_BASE_URL）：真实调用对齐与解析。

运行：python llm_smoke_test.py
"""

import sys

from state import build_state_from_alignment
from planner import apply_report, daily_blocks, init_plan, render_blocks

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


# 模拟 LLM 返回的对齐表草稿（结构与 llm.align_materials 的约定一致）
FAKE_ALIGNMENT = {
    "resources": ["视频源", "主教材", "习题库"],
    "knowledge_points": [
        {"name": "资料分析-增长率", "priority": 1.0, "confident": True, "items": [
            {"source": "视频源", "role": "learn", "title": "P1 增长率的概念与计算",
             "minutes": 25, "confident": True},
            {"source": "习题库", "role": "practice", "title": "增长率 基础题 30 道",
             "minutes": 20, "confident": True},
            {"source": "主教材", "role": "reference", "title": "第一章第一节 增长率",
             "minutes": 10, "confident": False},
        ]},
        {"name": "资料分析-比重", "priority": 0.9, "confident": False, "items": [
            {"source": "视频源", "role": "learn", "title": "P3 比重问题",
             "minutes": 25, "confident": True},
        ]},
    ],
}


def part_a():
    print("\n" + "=" * 64)
    print("Part A（离线）：结构链路 + 接口纪律")
    state = build_state_from_alignment(
        FAKE_ALIGNMENT, deadline="2026-12-20", life_profile="打工人",
        daily_total=120, start_date="2026-11-05"
    )
    ok(len(state["knowledge_points"]) == 2, "对齐表 → 状态：2 个知识点")
    ok(state["materials"]["resources"] == ["视频源", "主教材", "习题库"],
       "资源类型写入状态")
    result = init_plan(state)
    day1 = daily_blocks(state, result["first_day"])
    ok(len(day1) >= 1, "对齐表 → 状态 → 排程链路通")
    for line in render_blocks(day1):
        print("    首日：" + line)

    # 接口纪律：LLM 形状的解析结果必须能被 planner.apply_report 原样消费
    llm_shaped = {
        "outcome": "partial",
        "partial_ratio": 0.6,
        "too_hard_kps": ["kp1"],
        "too_heavy": False,
        "misaligned": False,
        "_text": "今天增长率太难，只看了六成",
    }
    summary = apply_report(state, result["first_day"], llm_shaped)
    ok(bool(summary["summary"]), "LLM 形状的解析结果被 apply_report 原样消费")
    kp1 = next(k for k in state["knowledge_points"] if k["id"] == "kp1")
    ok(kp1["status"] == "stuck", "『太难』→ kp1 标记卡点（真实 kp_id 生效）")
    for line in summary["summary"]:
        print("    Agent:", line)


def part_b():
    print("\n" + "=" * 64)
    print("Part B（联机）：真实调用 LLM")
    from llm import align_materials, parse_report, render_alignment
    pasted = """【B站视频合集：资料分析速成】
P1 增长率的概念与计算
P2 增长率的比较
P3 比重问题

【粉笔题库模块】
资料分析-增长率（基础题 30 道）
资料分析-比重（基础题 25 道）"""
    print("    1) align_materials...")
    alignment = align_materials(pasted)
    ok(bool(alignment["knowledge_points"]), "LLM 返回对齐表")
    print(render_alignment(alignment))

    state = build_state_from_alignment(
        alignment, deadline="2026-12-20", life_profile="打工人",
        daily_total=120, start_date="2026-11-05"
    )
    result = init_plan(state)
    day1 = daily_blocks(state, result["first_day"])
    slim = [{"kp_id": b["kp_id"], "kp_name": b["kp_name"], "title": b["title"]}
            for b in day1]
    parsed = parse_report("今天增长率太难了，只看了一半", slim)
    print(f"    2) parse_report：outcome={parsed['outcome']} "
          f"too_hard={parsed['too_hard_kps']} ratio={parsed.get('partial_ratio')}")
    ok(set(parsed) >= {"outcome", "too_hard_kps", "too_heavy", "misaligned", "_text"},
       "输出结构与占位版一致")
    summary = apply_report(state, result["first_day"], parsed)
    for line in summary["summary"]:
        print("    Agent:", line)


def main():
    print("llm.py 冒烟测试")
    part_a()

    import os
    import llm  # noqa: F401  导入即加载 .env
    if os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        try:
            part_b()
        except Exception as e:
            ok(False, f"Part B 联机失败：{e}")
    else:
        print("\n" + "=" * 64)
        print("Part B（联机）跳过：未配置 LLM_API_KEY / OPENAI_API_KEY。")
        print("在项目根目录建 .env 后重跑（模板见 .env.example）：")
        print("  LLM_API_KEY=你的key")
        print("  LLM_BASE_URL=https://api.deepseek.com/v1")
        print("  LLM_MODEL=deepseek-chat")

    print("\n" + "=" * 64)
    if FAILURES:
        print(f"结果：{len(FAILURES)}/{CHECKS} 项未通过")
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print(f"结果：全部通过（{CHECKS} 项检查）")


if __name__ == "__main__":
    main()
