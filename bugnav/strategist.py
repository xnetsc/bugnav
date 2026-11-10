"""LLM strategy advisor: the escape hatch beyond the fixed strategies.

The deterministic Bug algorithm handles the common cases. But a fixed
strategy set stays fixed forever - so at the moments where the fixed
logic is about to grind on blindly (repeated failures) or give up (full
circle on a large obstacle), the controller may CONSULT an LLM with a
structured situation report and receive back a bounded plan.

Safety contract (enforced HERE and in the controller, never by the LLM):
  * The LLM never emits coordinates. It may only:
      - try_angle: queue staging angles (degrees around the cluster)
      - adjust:    step angle / staging clearance, within hard bounds
      - abandon:   agree to give up this session
      - resolved:  claim the obstacle is gone/resolved (episode ends,
                   no success/abandon recorded - the robot just resumes)
  * At most 3 queued angles, at most 4 actions per plan.
  * Anything unparsable or out-of-vocabulary degrades to None, and None
    always means "do exactly what the fixed strategy would have done".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Hard bounds - the LLM can steer inside these, never outside.
STEP_RANGE = (10.0, 90.0)
CLEARANCE_RANGE = (0.30, 1.00)
MAX_ANGLES = 3
MAX_ACTIONS = 4


@dataclass
class Plan:
    angles: list[float] = field(default_factory=list)
    step_angle_deg: float | None = None
    staging_clearance_m: float | None = None
    abandon: bool = False
    resolved: bool = False
    reason: str = ""


def _clamp(v: float, rng: tuple[float, float]) -> float:
    return max(rng[0], min(rng[1], v))


def parse_plan(text: str) -> Plan | None:
    """Parse an LLM reply into a validated Plan, or None if unusable."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("actions"), list):
        return None
    plan = Plan(reason=str(raw.get("reason", ""))[:300])
    for act in raw["actions"][:MAX_ACTIONS]:
        if not isinstance(act, dict):
            continue
        kind = act.get("type")
        if kind == "try_angle":
            try:
                theta = float(act["theta_deg"]) % 360.0
            except (KeyError, TypeError, ValueError):
                continue
            if len(plan.angles) < MAX_ANGLES:
                plan.angles.append(theta)
        elif kind == "adjust":
            if "step_angle_deg" in act:
                try:
                    plan.step_angle_deg = _clamp(
                        float(act["step_angle_deg"]), STEP_RANGE)
                except (TypeError, ValueError):
                    pass
            if "staging_clearance_m" in act:
                try:
                    plan.staging_clearance_m = _clamp(
                        float(act["staging_clearance_m"]), CLEARANCE_RANGE)
                except (TypeError, ValueError):
                    pass
        elif kind == "abandon":
            plan.abandon = True
        elif kind == "resolved":
            plan.resolved = True
    if not (plan.angles or plan.step_angle_deg or plan.staging_clearance_m
            or plan.abandon or plan.resolved):
        return None
    return plan


SYSTEM = """你是一个扫地机器人的策略顾问。机器人被一个障碍物挡住，外置程序正在用固定策略（退回入口、沿障碍物轮廓每 30° 换一个接近角度尝试）处理，但在某个决策点需要你的判断。

你只能输出一个 JSON 对象，不要输出任何其他文字。格式：
{"actions": [...], "reason": "一句话说明"}

允许的 action（最多 4 个）：
- {"type": "try_angle", "theta_deg": 数字} —— 让机器人接下来优先从障碍簇中心极角 theta_deg 方向的集结点尝试接近（最多给 3 个，按优先级排序）。结合已扫过的角度、撞击点分布和障碍形状推测开口最可能在哪一侧。
- {"type": "adjust", "step_angle_deg": 10~90 的数字, "staging_clearance_m": 0.3~1.0 的数字} —— 调整后续步进角度/集结环间距（可选字段）。
- {"type": "abandon"} —— 你也判断本轮无解，同意放弃该区域（仅本次任务）。
- {"type": "resolved"} —— 你判断障碍可能已移走/问题已解除，让机器人直接恢复全屋清扫。

你没有坐标发明权：不要输出任何坐标，只能给角度与参数。reason 用简体中文。"""


def build_messages(situation: dict) -> list[dict]:
    """Turn the controller's situation dict into chat messages."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "当前局面（JSON）：\n"
         + json.dumps(situation, ensure_ascii=False, indent=1)},
    ]
