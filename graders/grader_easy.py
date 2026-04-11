"""
Grader for Task 1: License Classification

Scoring dimensions:
  - SPDX accuracy (65%)      : exact canonical SPDX identifier match
  - Category accuracy (25%)  : license category correct even if SPDX alias used
  - Brier calibration (10%)  : proper-scoring-rule penalty on confidence×correctness
                               Replaces naive "high-conf + correct = bonus" —
                               agents that say 0.99 and are wrong are penalised
                               more than agents that say 0.5 and are wrong.
  - Coverage multiplier      : penalises skipping files
  - Efficiency bonus (+5%)   : completes under 60% of step budget

Brier score detail
──────────────────
For each file, Brier contribution = (confidence - correctness)²
where correctness = 1.0 if SPDX correct else 0.0.
Mean Brier score ∈ [0, 1] where 0 = perfect calibration.
We convert to a reward: calibration_reward = 1 - mean_brier, scaled to 0.10 weight.
An agent that always says 0.5 with mixed results scores ~0.75 on calibration.
An agent that says 0.99 and is always right scores ~0.99.
An agent that says 0.99 and is always wrong scores ~0.02 — heavy penalty.

Deceptive license handling
──────────────────────────
Some scenarios inject files where the SPDX-License-Identifier header
deliberately uses a common alias (e.g. "GPLv3") while the body text contains
contradictory permissive-sounding language. These are flagged in ground_truth
with "is_deceptive": True. The agent gets a +0.03 bonus per deceptive file
it correctly resolves, rewarding careful reading over header-skimming.
"""
from __future__ import annotations
from env.models import ScanFinding, Reward
from data.license_corpus import LICENSE_CORPUS


SPDX_WEIGHT      = 0.65
CATEGORY_WEIGHT  = 0.25
CALIBRATION_WEIGHT = 0.10  # Brier-based, replaces naive confidence bonus
DECEPTIVE_BONUS  = 0.03    # per correctly resolved deceptive file


def _normalize_spdx(raw: str) -> str:
    """Map SPDX aliases to canonical identifiers."""
    raw = raw.strip()
    for spdx_id, meta in LICENSE_CORPUS.items():
        if raw == spdx_id:
            return spdx_id
        if raw.lower() in [a.lower() for a in meta.get("aliases", [])]:
            return spdx_id
    return raw


def score(
    findings: list[ScanFinding],
    ground_truth: dict[str, dict],
    steps_used: int,
    max_steps: int,
) -> Reward:
    if not ground_truth:
        return Reward(total=0.0, message="No ground truth available")

    n = len(ground_truth)
    agent_map: dict[str, ScanFinding] = {f.source_id: f for f in findings}

    spdx_correct     = 0
    category_correct = 0
    brier_sum        = 0.0
    brier_count      = 0
    deceptive_bonus  = 0.0

    for file_id, truth in ground_truth.items():
        finding = agent_map.get(file_id)
        if finding is None:
            # Unattempted file: treat as wrong with neutral confidence 0.5
            brier_sum   += (0.5 - 0.0) ** 2
            brier_count += 1
            continue

        agent_spdx   = _normalize_spdx(finding.license_spdx or "")
        true_spdx    = truth["spdx_id"]
        true_category = truth["category"]

        spdx_ok = (agent_spdx == true_spdx)
        cat_ok  = (finding.category and finding.category.value == true_category)

        if spdx_ok:
            spdx_correct += 1
        if cat_ok:
            category_correct += 1

        # Brier score: (predicted_prob - outcome)²
        conf       = max(0.0, min(1.0, finding.agent_confidence))
        correctness = 1.0 if spdx_ok else 0.0
        brier_sum   += (conf - correctness) ** 2
        brier_count += 1

        # Deceptive file bonus
        if truth.get("is_deceptive") and spdx_ok:
            deceptive_bonus += DECEPTIVE_BONUS

    coverage = len(agent_map) / n

    spdx_score  = (spdx_correct / n) * SPDX_WEIGHT
    cat_score   = (category_correct / n) * CATEGORY_WEIGHT

    # Brier calibration: 0 = perfect, 1 = worst. Convert to reward.
    mean_brier        = brier_sum / max(brier_count, 1)
    calibration_score = (1.0 - mean_brier) * CALIBRATION_WEIGHT

    raw = (spdx_score + cat_score + calibration_score) * coverage + deceptive_bonus

    # Efficiency bonus
    efficiency_bonus = 0.0
    if steps_used <= int(max_steps * 0.6) and raw >= 0.5:
        efficiency_bonus = 0.05 * (1 - steps_used / max_steps)

    total = min(1.0, raw + efficiency_bonus)

    return Reward(
        total=round(total, 4),
        breakdown={
            "spdx_accuracy":       round(spdx_score, 4),
            "category_accuracy":   round(cat_score, 4),
            "brier_calibration":   round(calibration_score, 4),
            "mean_brier_score":    round(mean_brier, 4),
            "coverage":            round(coverage, 4),
            "deceptive_bonus":     round(deceptive_bonus, 4),
            "efficiency_bonus":    round(efficiency_bonus, 4),
        },
        message=(
            f"Classified {len(agent_map)}/{n} files. "
            f"SPDX correct: {spdx_correct}/{n}, "
            f"Category correct: {category_correct}/{n}, "
            f"Mean Brier: {mean_brier:.3f}"
        ),
    )