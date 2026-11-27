"""A/B test statistics: Wilson CIs, two-proportion z-test, Bayesian P(best), sample sizing."""
from __future__ import annotations

import math
import random
from statistics import NormalDist

_N = NormalDist()


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def two_prop_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float] | None:
    """Return (z, two-sided p) for H0: p1 == p2, or None if undefined."""
    if n1 <= 0 or n2 <= 0:
        return None
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (k2 / n2 - k1 / n1) / se
    return z, 2 * (1 - _N.cdf(abs(z)))


def prob_best(arms: list[tuple[int, int]], draws: int = 20000, seed: int = 42) -> list[float]:
    """Monte-Carlo P(arm has the highest true rate) under Beta(1,1) priors."""
    rng = random.Random(seed)
    wins = [0] * len(arms)
    for _ in range(draws):
        samples = [rng.betavariate(1 + k, 1 + max(0, n - k)) for k, n in arms]
        wins[samples.index(max(samples))] += 1
    return [w / draws for w in wins]


def required_sample_size(baseline: float, relative_mde: float,
                         alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm sample size to detect a relative lift with a two-sided two-proportion test."""
    if not (0 < baseline < 1):
        raise ValueError("baseline must be between 0 and 1 (exclusive)")
    if relative_mde <= 0:
        raise ValueError("relative_mde must be > 0")
    p1 = baseline
    p2 = baseline * (1 + relative_mde)
    if p2 >= 1:
        raise ValueError("baseline * (1 + relative_mde) must be < 1")
    z_a = _N.inv_cdf(1 - alpha / 2)
    z_b = _N.inv_cdf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)


def summarize_experiment(arms: list[dict], min_per_arm: int = 1000,
                         win_prob: float = 0.95, alpha: float = 0.05) -> dict:
    """arms: [{id,label,impressions,conversions,is_control}] -> per-arm stats + verdict."""
    control = next((a for a in arms if a["is_control"]), arms[0])
    probs = prob_best([(a["conversions"], a["impressions"]) for a in arms])
    n_comp = max(1, len(arms) - 1)
    out = []
    for a, pb in zip(arms, probs):
        n, k = a["impressions"], a["conversions"]
        rate = k / n if n else None
        ci = wilson_interval(k, n)
        row = {**a, "rate": rate, "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None,
               "prob_best": pb, "lift_vs_control": None, "p_value": None, "p_value_adj": None}
        if a is not control and n and control["impressions"]:
            crate = control["conversions"] / control["impressions"]
            if crate > 0:
                row["lift_vs_control"] = (rate - crate) / crate
            t = two_prop_ztest(control["conversions"], control["impressions"], k, n)
            if t:
                row["p_value"] = t[1]
                row["p_value_adj"] = min(1.0, t[1] * n_comp)   # Bonferroni
        out.append(row)

    if any(a["impressions"] < min_per_arm for a in arms):
        status, winner = "insufficient_data", None
    else:
        best = max(out, key=lambda r: r["prob_best"])
        significant = best["is_control"] or (best["p_value_adj"] is not None and best["p_value_adj"] < alpha)
        if best["prob_best"] >= win_prob and significant:
            status, winner = "winner", best["id"]
        else:
            status, winner = "inconclusive", None
    return {"arms": out, "status": status, "winner_arm_id": winner,
            "min_impressions_per_arm": min_per_arm}
