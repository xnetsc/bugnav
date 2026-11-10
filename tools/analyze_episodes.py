"""Offline analysis of bugnav recordings, with optional LLM assistance.

Reads a bugnav data directory (samples.jsonl + events.jsonl as written by
bugnav.runner), computes a deterministic digest (cleaning time, distance,
phase split, error histogram, would-send commands, hot cells), and - when
an LLM is configured via environment (see bugnav/llm.py) - asks it for a
narrative recap: what likely happened, what the obstacles probably were,
whether the detector thresholds look sane, what to tune.

Usage:
  python tools/analyze_episodes.py [data_dir] [-o report.md]

Without BUGNAV_LLM_API_KEY the report contains the digest only.
"""

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ERROR_NAMES = {
    102: "悬空 ERROR_HANG",
    103: "驱动轮异常",
    105: "被困 ERROR_STUCK",
    204: "驱动轮悬空 ERROR_WHEEL_HANG",
}


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def digest(samples, events) -> dict:
    d = {"samples": len(samples)}
    if not samples:
        return d
    t0, t1 = samples[0]["t"], samples[-1]["t"]
    d["span_s"] = round(t1 - t0, 1)
    dist = 0.0
    for a, b in zip(samples, samples[1:]):
        dist += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    d["distance_m"] = round(dist, 2)
    d["states"] = dict(Counter(s["state"] for s in samples))
    d["phases"] = dict(Counter(s.get("phase", "?") for s in samples))
    errs = Counter(s["error"] for s in samples if s.get("error"))
    d["errors"] = {
        f"{code} {ERROR_NAMES.get(code, '')}".strip(): n
        for code, n in errs.most_common()
    }
    cells = Counter(
        (round(s["x"] * 2) / 2, round(s["y"] * 2) / 2) for s in samples)
    d["hot_cells"] = [
        {"cell": list(c), "samples": n} for c, n in cells.most_common(5)]
    xs = [s["x"] for s in samples]
    ys = [s["y"] for s in samples]
    d["bbox_m"] = [round(min(xs), 2), round(min(ys), 2),
                   round(max(xs), 2), round(max(ys), 2)]
    would = Counter()
    notifies = []
    for e in events:
        msg = e.get("msg", "")
        if "[observe] would send" in msg:
            would[msg.split("would send", 1)[1].strip().split(" ")[0]] += 1
        elif "NOTIFY:" in msg:
            notifies.append(msg.split("NOTIFY:", 1)[1].strip())
    d["would_send"] = dict(would)
    d["notifies"] = notifies[:20]
    # Struggle clusters: samples in non-monitor phases, grouped by time gaps
    episodes = []
    cur = []
    for s in samples:
        if s.get("phase") not in (None, "monitor"):
            if cur and s["t"] - cur[-1]["t"] > 60:
                episodes.append(cur)
                cur = []
            cur.append(s)
    if cur:
        episodes.append(cur)
    d["episodes"] = [
        {
            "start_t": round(ep[0]["t"] - t0, 1),
            "duration_s": round(ep[-1]["t"] - ep[0]["t"], 1),
            "centre": [round(statistics.mean(s["x"] for s in ep), 2),
                       round(statistics.mean(s["y"] for s in ep), 2)],
            "phases": dict(Counter(s["phase"] for s in ep)),
        }
        for ep in episodes
    ]
    return d


PROMPT = """你是扫地机器人运行日志的分析助手。下面是外置避障程序 bugnav 记录的一段数据的确定性统计摘要（JSON）。请用简体中文写一份简短复盘，包含：
1. 这段时间机器人整体在干什么（清扫覆盖、时长、距离是否合理）；
2. 逐个复盘每个非 monitor 阶段的 episode：最可能遇到了什么（结合中心坐标、持续时长、阶段构成推测障碍物类型，如椅子腿、桌腿、U 形底座、门槛），处理结果像是成功绕过还是放弃；
3. 错误码（如有）说明什么；
4. 检测器与控制参数的调参建议：触发是否太灵敏/太迟钝、集结环/步进角度/挣扎预算是否合理，给出具体数值方向。
只基于数据说话，数据不足处明说不足，不要编造。

数据摘要：
"""


async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    samples = load_jsonl(os.path.join(args.data_dir, "samples.jsonl"))
    events = load_jsonl(os.path.join(args.data_dir, "events.jsonl"))
    d = digest(samples, events)

    parts = ["# bugnav 运行分析\n",
             "## 确定性摘要\n",
             "```json",
             json.dumps(d, ensure_ascii=False, indent=1),
             "```\n"]

    from bugnav import llm

    if llm.llm_configured():
        try:
            text = await llm.chat([
                {"role": "user",
                 "content": PROMPT + json.dumps(d, ensure_ascii=False)},
            ])
            parts += ["## LLM 辅助复盘\n", text, ""]
        except Exception as exc:
            parts += ["## LLM 辅助复盘\n", f"（LLM 调用失败：{exc}）", ""]
    else:
        parts += ["## LLM 辅助复盘\n",
                  "（未配置 BUGNAV_LLM_API_KEY，跳过。设置该环境变量后重跑即可。）",
                  ""]

    out = args.out or os.path.join(args.data_dir, "analysis.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"report written: {out} (samples={d.get('samples', 0)}, "
          f"episodes={len(d.get('episodes', []))})")


if __name__ == "__main__":
    asyncio.run(amain())
