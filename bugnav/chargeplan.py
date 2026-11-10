"""Charge planning: minimize TOTAL elapsed time of a cleaning bout.

Objective (user-settled): cleaning time + charging time + per-charge
overheads, summed, must be minimal - a 07:00 task must not still be
running at 20:00 because it kept charging to a fixed threshold.

Physics that drives the optimum:
  * Charging is nonlinear: fast at low state-of-charge, slow in the
    top taper. Charging "just enough" is cheap; the last 20% is dear.
  * Every dock cycle costs fixed overhead (travel back, mop wash,
    repositioning) on top of the charge itself.
  * So: charge exactly the energy the remaining area needs (plus a
    safety margin), unless a bit more now provably eliminates a whole
    extra cycle later. The solver enumerates candidate resume levels
    and simulates the rest of the bout for each, using LEARNED rates:
      - drain: battery % per m^2 cleaned (from this home's history)
      - charge rate per SOC band (% per minute while docked)
      - cycle overhead seconds (dock/resume transitions)
    All inputs carry defaults until enough real data is recorded.

The deterministic solver proposes; the LLM supervisor reviews the
proposal with context the solver cannot see (time of day, whether the
remaining scrap of area is worth another cycle at all) and may
override the resume level within bounds or stop the session instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# SOC bands for the charge-rate curve (%/min). Defaults are typical
# Li-ion robot behaviour; learned values replace them over time.
DEFAULT_BAND_RATES = [(50, 1.6), (80, 1.0), (100, 0.45)]
DEFAULT_DRAIN_PCT_PER_M2 = 0.55  # ~180 m^2 per full charge
DEFAULT_OVERHEAD_S = 300.0  # travel + mop wash + resume per cycle
MIN_RESUME_SOC = 15  # firmware will not start cleaning below ~this
MARGIN = 1.25  # safety factor on the energy estimate


@dataclass
class ChargeModel:
    band_rates: list[tuple[int, float]] = field(
        default_factory=lambda: list(DEFAULT_BAND_RATES))
    drain_pct_per_m2: float = DEFAULT_DRAIN_PCT_PER_M2
    overhead_s: float = DEFAULT_OVERHEAD_S

    def charge_minutes(self, from_soc: float, to_soc: float) -> float:
        """Minutes to charge between two SOC levels (band-integrated)."""
        if to_soc <= from_soc:
            return 0.0
        minutes = 0.0
        soc = from_soc
        for band_top, rate in self.band_rates:
            if soc >= band_top:
                continue
            seg = min(to_soc, band_top) - soc
            if seg > 0:
                minutes += seg / rate
                soc += seg
            if soc >= to_soc:
                break
        return minutes

    def usable_m2(self, soc: float, reserve: float = 8.0) -> float:
        """Area cleanable starting at `soc` before hitting reserve."""
        return max(0.0, (soc - reserve) / self.drain_pct_per_m2)


def plan_resume(now_soc: float, remaining_m2: float,
                model: ChargeModel) -> dict:
    """Pick the resume SOC minimizing predicted remaining wall time.

    Simulates the rest of the bout for each candidate first-charge
    target: clean what the charge buys, loop (charge again with the
    same policy shape: subsequent charges go to min(100, need)) until
    the remaining area is done. Returns the best plan.
    """
    need_pct = remaining_m2 * model.drain_pct_per_m2 * MARGIN
    base = max(MIN_RESUME_SOC, now_soc)
    candidates = sorted({
        round(min(100.0, max(base, now_soc + need_pct * f)))
        for f in (0.85, 1.0, 1.15, 1.3)
    } | {100})
    best: dict | None = None
    for target in candidates:
        t_min = 0.0
        soc = float(now_soc)
        remaining = float(remaining_m2)
        charges = 0
        guard = 0
        while remaining > 0.5 and guard < 8:
            guard += 1
            want = target if charges == 0 else min(
                100.0, soc + remaining * model.drain_pct_per_m2 * MARGIN)
            want = max(want, MIN_RESUME_SOC)
            if want > soc:
                t_min += model.charge_minutes(soc, want)
                t_min += model.overhead_s / 60.0
                charges += 1
                soc = want
            cover = model.usable_m2(soc)
            if cover <= 0:
                break
            cleaned = min(cover, remaining)
            remaining -= cleaned
            soc -= cleaned * model.drain_pct_per_m2
            t_min += cleaned / 1.1  # ~1.1 m^2/min cleaning speed
        total = t_min if remaining <= 0.5 else float("inf")
        if best is None or total < best["est_total_min"]:
            best = {
                "resume_soc": int(target),
                "est_total_min": round(total, 1),
                "est_charges": charges,
                "remaining_m2": round(remaining_m2, 1),
                "need_pct_with_margin": round(need_pct, 1),
            }
    assert best is not None
    return best
