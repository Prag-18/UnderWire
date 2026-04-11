"""
Grader for Task 2: Dependency Conflict Detection
Scores on precision/recall of conflict flagging.
"""
from __future__ import annotations
from typing import Any

from env.models import ScanFinding, ConflictSeverity, Reward


def score(
    findings: list[ScanFinding],
    ground_truth: dict[str, dict],
    steps_used: int,
    max_steps: int,
) -> Reward:
    true_conflicts = {
        dep_id for dep_id, info in ground_truth.items() if info["is_conflict"]
    }
    true_clears = {
        dep_id for dep_id, info in ground_truth.items() if not info["is_conflict"]
    }

    flagged: dict[str, ScanFinding] = {
        f.source_id: f for f in findings if f.severity != ConflictSeverity.NONE
    }
    cleared: set[str] = {
        f.source_id for f in findings if f.severity == ConflictSeverity.NONE
    }

    tp = len(flagged.keys() & true_conflicts)
    fp = len(flagged.keys() & true_clears)
    fn = len(true_conflicts - flagged.keys())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    # Severity accuracy bonus — for each TP did agent get severity right?
    severity_score = 0.0
    for dep_id, finding in flagged.items():
        if dep_id in true_conflicts:
            true_sev = ground_truth[dep_id]["conflict_severity"]
            agent_sev = finding.severity.value if finding.severity else "none"
            if agent_sev == true_sev:
                severity_score += 1.0
            elif abs(["none","low","medium","high","critical"].index(agent_sev)
                     - ["none","low","medium","high","critical"].index(true_sev)) == 1:
                severity_score += 0.5  # off-by-one severity: partial credit

    severity_bonus = (severity_score / max(len(true_conflicts), 1)) * 0.2

    # FP penalty: false alarms waste engineering time
    fp_penalty = min(0.3, fp * 0.05)

    raw = f1 * 0.7 + severity_bonus - fp_penalty

    # Efficiency
    efficiency_bonus = 0.0
    if steps_used <= int(max_steps * 0.5) and raw >= 0.5:
        efficiency_bonus = 0.05

    total = max(0.0, min(1.0, raw + efficiency_bonus))

    return Reward(
        total=round(total, 4),
        breakdown={
            "precision":        round(precision, 4),
            "recall":           round(recall, 4),
            "f1":               round(f1, 4),
            "severity_bonus":   round(severity_bonus, 4),
            "fp_penalty":       round(-fp_penalty, 4),
            "efficiency_bonus": round(efficiency_bonus, 4),
            "tp": tp, "fp": fp, "fn": fn,
        },
        message=(
            f"TP={tp}, FP={fp}, FN={fn}. "
            f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}"
        ),
    )