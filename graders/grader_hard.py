"""
Grader for Task 3: Compliance Report Generation

Scoring dimensions and intentional design ceilings
───────────────────────────────────────────────────
Component              Weight   Notes
─────────────────────────────────────────────────────────────────────────
Critical recall         0.20    Recall on AGPL/GPL in SaaS project
High recall             0.10    Recall on GPL-2.0 (non-SaaS conflicts)
Report completeness     0.25    Count accuracy, actions, risk, summary
Remediation quality     0.15    Substantive remediation per critical dep
Unknown file handling   0.05    Flags files with unidentifiable licenses
FP penalty            ≤−0.20    −0.04 per false positive conflict
Efficiency bonus       ≤+0.05   Under 60% step budget: +0.05, under 80%: +0.025

INTENTIONAL ORACLE CEILING: ~0.80 (not 1.0)
────────────────────────────────────────────
The hard task deliberately cannot be fully solved in a single pass because:

1. unknown_handling (0.05 max) requires flagging 3 vendor files whose licenses
   (OpenSSL, IJG, mixed) have no canonical SPDX identifier. A correct agent
   should flag them for manual legal review, not guess an SPDX ID. The grader
   only awards this if the file IS flagged as unknown — an agent that guesses
   MIT on every unknown file scores 0 here, which is the correct penalty.

2. report completeness (0.25 max) requires knowing the exact violation count.
   The ground truth has a specific integer; agents that estimate off by more
   than 1 receive partial credit only.

3. The scenario has 26 deps + 6 files (32 total). Even a perfect agent needs
   ~35 steps to process all items plus generate a report, leaving little room
   for the efficiency bonus unless the agent prioritises critical items first.

This ceiling is intentional: it distinguishes "good agents that understand
compliance" from "perfect oracle agents with access to ground truth". An agent
scoring 0.70+ on the hard task has genuinely learned legal-engineering reasoning.
"""
from __future__ import annotations
from typing import Any

from env.models import ScanFinding, ComplianceReport, ConflictSeverity, Reward


_SEV_ORDER = ["none", "low", "medium", "high", "critical"]


def _sev_idx(s: str) -> int:
    try:
        return _SEV_ORDER.index(s.lower())
    except ValueError:
        return 0


def score(
    findings: list[ScanFinding],
    report: ComplianceReport | None,
    ground_truth: dict[str, Any],
    steps_used: int,
    max_steps: int,
) -> Reward:
    breakdown: dict[str, float] = {}

    # ── 1. Critical finding detection (0.0–0.30) ─────────────────
    expected_critical: list[str] = ground_truth.get("expected_critical", [])
    expected_high: list[str]     = ground_truth.get("expected_high", [])

    flagged_critical = {
        f.source_id for f in findings if f.severity == ConflictSeverity.CRITICAL
    }
    flagged_high = {
        f.source_id for f in findings if f.severity == ConflictSeverity.HIGH
    }

    critical_recall = (
        len(flagged_critical & set(expected_critical)) / max(len(expected_critical), 1)
    )
    high_recall = (
        len(flagged_high & set(expected_high)) / max(len(expected_high), 1)
    )
    finding_score = critical_recall * 0.20 + high_recall * 0.10
    breakdown["critical_recall"] = round(critical_recall, 4)
    breakdown["high_recall"]     = round(high_recall, 4)
    breakdown["finding_score"]   = round(finding_score, 4)

    # ── 2. False positive penalty (up to −0.20) ───────────────────
    deps_truth = ground_truth.get("deps", {})
    fp_count = sum(
        1 for f in findings
        if f.source_id in deps_truth
        and deps_truth[f.source_id]["severity"] == "none"
        and f.severity != ConflictSeverity.NONE
    )
    fp_penalty = min(0.20, fp_count * 0.04)
    breakdown["fp_penalty"] = round(-fp_penalty, 4)

    # ── 3. Report structure & completeness (0.0–0.25) ─────────────
    # Intentional ceiling: partial credit for off-by-one violation count,
    # zero credit for counts that are wrong by more than 50%.
    report_score = 0.0
    if report:
        total_violations = ground_truth.get("total_violations", 0)

        count_err = abs(report.violation_count - total_violations)
        count_acc = max(0.0, 1.0 - count_err / max(total_violations, 1))
        report_score += count_acc * 0.10

        if report.recommended_actions and len(report.recommended_actions) >= 2:
            report_score += 0.07

        if expected_critical and report.overall_risk == ConflictSeverity.CRITICAL:
            report_score += 0.05
        elif not expected_critical and expected_high and report.overall_risk == ConflictSeverity.HIGH:
            report_score += 0.03

        if report.summary and len(report.summary) > 50:
            report_score += 0.03

    breakdown["report_score"] = round(report_score, 4)

    # ── 4. Remediation quality (0.0–0.15) ─────────────────────────
    remediation_score = 0.0
    must_remediate: list[str] = ground_truth.get("must_remediate", [])
    remediated_ids = {
        f.source_id for f in findings
        if f.source_id in must_remediate
        and f.remediation
        and len(f.remediation) > 20
    }
    if must_remediate:
        remediation_score = len(remediated_ids) / len(must_remediate) * 0.15
    breakdown["remediation_score"] = round(remediation_score, 4)

    # ── 5. Unknown file handling (0.0–0.05) ───────────────────────
    # Intentional ceiling: agents that guess a SPDX ID for unknown vendor
    # files score 0 here. Only flagging them as UNKNOWN earns credit.
    unknown_files: list[str] = ground_truth.get("expected_unknown_files", [])
    flagged_unknown = {
        f.source_id for f in findings
        if f.source_id in unknown_files
        and f.category is not None
        and f.category.value == "unknown"
    }
    unknown_score = len(flagged_unknown) / max(len(unknown_files), 1) * 0.05
    breakdown["unknown_handling"] = round(unknown_score, 4)

    # ── 6. Efficiency bonus (0.0–0.05) ────────────────────────────
    efficiency_bonus = 0.0
    if steps_used <= int(max_steps * 0.6):
        efficiency_bonus = 0.05
    elif steps_used <= int(max_steps * 0.8):
        efficiency_bonus = 0.025
    breakdown["efficiency_bonus"] = round(efficiency_bonus, 4)

    # ── Total ──────────────────────────────────────────────────────
    raw = (
        finding_score
        + report_score
        + remediation_score
        + unknown_score
        - fp_penalty
        + efficiency_bonus
    )
    total = max(0.0, min(1.0, raw))

    return Reward(
        total=round(total, 4),
        breakdown=breakdown,
        message=(
            f"Critical recall={critical_recall:.2f}, "
            f"High recall={high_recall:.2f}, "
            f"FPs={fp_count}, "
            f"Report={'present' if report else 'missing'}, "
            f"Steps={steps_used}/{max_steps}"
        ),
    )