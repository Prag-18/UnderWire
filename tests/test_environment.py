"""
Full test suite for the License Compliance Scanner OpenEnv environment.
Tests reset/step/state API, reward signals, and all three graders.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import LicenseComplianceEnv
from env.models import (
    Action, ActionType, LicenseCategory, ConflictSeverity,
    ComplianceReport, ScanFinding
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    env = LicenseComplianceEnv(task_id="classify_licenses", seed=42)
    env.reset()
    return env

@pytest.fixture
def medium_env():
    env = LicenseComplianceEnv(task_id="detect_conflicts", seed=42)
    env.reset()
    return env

@pytest.fixture
def hard_env():
    env = LicenseComplianceEnv(task_id="generate_compliance_report", seed=42)
    env.reset()
    return env


# ──────────────────────────────────────────────────────────────────
# API contract tests
# ──────────────────────────────────────────────────────────────────

class TestOpenEnvAPI:
    def test_reset_returns_observation(self):
        env = LicenseComplianceEnv(task_id="classify_licenses", seed=42)
        obs = env.reset()
        assert obs.task_id == "classify_licenses"
        assert obs.step == 0
        assert obs.done is False
        assert obs.max_steps > 0

    def test_reset_populates_files(self):
        env = LicenseComplianceEnv(task_id="classify_licenses", seed=42)
        obs = env.reset()
        assert len(obs.files_to_scan) > 0, "Easy task must have files to classify"

    def test_reset_populates_deps(self):
        env = LicenseComplianceEnv(task_id="detect_conflicts", seed=42)
        obs = env.reset()
        assert len(obs.dependencies) > 0, "Medium task must have dependencies"

    def test_step_increments_step_counter(self, easy_env):
        action = Action(
            action_type=ActionType.CLASSIFY_LICENSE,
            target_id=easy_env._files[0].file_id,
            classification="MIT",
            category=LicenseCategory.PERMISSIVE,
            confidence=0.9,
        )
        obs, reward, done, info = easy_env.step(action)
        assert obs.step == 1
        assert info["step"] == 1

    def test_step_returns_typed_reward(self, easy_env):
        action = Action(
            action_type=ActionType.CLASSIFY_LICENSE,
            target_id=easy_env._files[0].file_id,
            classification="MIT",
            category=LicenseCategory.PERMISSIVE,
            confidence=0.9,
        )
        _, reward, _, _ = easy_env.step(action)
        assert 0.0 <= reward.total <= 1.0
        assert isinstance(reward.breakdown, dict)
        assert isinstance(reward.message, str)

    def test_state_returns_full_state(self, easy_env):
        state = easy_env.state()
        assert state.task_id == "classify_licenses"
        assert state.step == 0
        assert state.ground_truth is not None
        assert isinstance(state.ground_truth, dict)

    def test_step_after_done_raises(self, easy_env):
        easy_env._done = True
        with pytest.raises(RuntimeError, match="Episode is done"):
            action = Action(action_type=ActionType.MARK_REVIEWED, target_id="x")
            easy_env.step(action)

    def test_reset_clears_findings(self, easy_env):
        action = Action(
            action_type=ActionType.CLASSIFY_LICENSE,
            target_id=easy_env._files[0].file_id,
            classification="MIT",
            category=LicenseCategory.PERMISSIVE,
        )
        easy_env.step(action)
        assert len(easy_env._findings) > 0
        easy_env.reset()
        assert len(easy_env._findings) == 0

    def test_deterministic_with_same_seed(self):
        env1 = LicenseComplianceEnv(task_id="classify_licenses", seed=99)
        env2 = LicenseComplianceEnv(task_id="classify_licenses", seed=99)
        obs1 = env1.reset()
        obs2 = env2.reset()
        ids1 = [f.file_id for f in obs1.files_to_scan]
        ids2 = [f.file_id for f in obs2.files_to_scan]
        assert ids1 == ids2

    def test_different_seeds_give_different_scenarios(self):
        env1 = LicenseComplianceEnv(task_id="classify_licenses", seed=1)
        env2 = LicenseComplianceEnv(task_id="classify_licenses", seed=2)
        obs1 = env1.reset()
        obs2 = env2.reset()
        snippets1 = [f.content_snippet for f in obs1.files_to_scan]
        snippets2 = [f.content_snippet for f in obs2.files_to_scan]
        assert snippets1 != snippets2


# ──────────────────────────────────────────────────────────────────
# Task 1 (Easy) tests
# ──────────────────────────────────────────────────────────────────

class TestEasyTask:
    def test_correct_classification_earns_reward(self, easy_env):
        first_file = easy_env._files[0]
        true_spdx = easy_env._ground_truth[first_file.file_id]["spdx_id"]
        true_cat  = easy_env._ground_truth[first_file.file_id]["category"]

        action = Action(
            action_type=ActionType.CLASSIFY_LICENSE,
            target_id=first_file.file_id,
            classification=true_spdx,
            category=LicenseCategory(true_cat),
            confidence=0.95,
        )
        _, reward, _, _ = easy_env.step(action)
        assert reward.total > 0.0

    def test_wrong_classification_earns_less(self, easy_env):
        first_file = easy_env._files[0]
        true_spdx = easy_env._ground_truth[first_file.file_id]["spdx_id"]
        wrong_spdx = "GPL-3.0-only" if true_spdx != "GPL-3.0-only" else "MIT"

        action = Action(
            action_type=ActionType.CLASSIFY_LICENSE,
            target_id=first_file.file_id,
            classification=wrong_spdx,
            category=LicenseCategory.COPYLEFT_STRONG,
            confidence=0.5,
        )
        _, reward_wrong, _, _ = easy_env.step(action)

        easy_env.reset()
        action_correct = Action(
            action_type=ActionType.CLASSIFY_LICENSE,
            target_id=first_file.file_id,
            classification=true_spdx,
            category=LicenseCategory(easy_env._ground_truth[first_file.file_id]["category"]),
            confidence=0.95,
        )
        _, reward_correct, _, _ = easy_env.step(action_correct)
        assert reward_correct.total >= reward_wrong.total

    def test_classifying_all_files_ends_episode(self, easy_env):
        done = False
        for f in easy_env._files:
            truth = easy_env._ground_truth.get(f.file_id, {})
            action = Action(
                action_type=ActionType.CLASSIFY_LICENSE,
                target_id=f.file_id,
                classification=truth.get("spdx_id", "MIT"),
                category=LicenseCategory(truth.get("category", "permissive")),
                confidence=0.9,
            )
            _, _, done, _ = easy_env.step(action)
        assert done is True

    def test_final_score_range(self, easy_env):
        for f in easy_env._files:
            action = Action(
                action_type=ActionType.CLASSIFY_LICENSE,
                target_id=f.file_id,
                classification="MIT",
                category=LicenseCategory.PERMISSIVE,
                confidence=0.7,
            )
            easy_env.step(action)
        score = easy_env.final_score()
        assert 0.0 <= score.total <= 1.0

    def test_perfect_agent_scores_high(self, easy_env):
        for f in easy_env._files:
            truth = easy_env._ground_truth.get(f.file_id, {})
            action = Action(
                action_type=ActionType.CLASSIFY_LICENSE,
                target_id=f.file_id,
                classification=truth.get("spdx_id", "MIT"),
                category=LicenseCategory(truth.get("category", "permissive")),
                confidence=0.95,
            )
            easy_env.step(action)
        score = easy_env.final_score()
        assert score.total >= 0.7, f"Perfect agent should score >= 0.7, got {score.total}"


# ──────────────────────────────────────────────────────────────────
# Task 2 (Medium) tests
# ──────────────────────────────────────────────────────────────────

class TestMediumTask:
    def test_flagging_true_conflict_earns_reward(self, medium_env):
        # Find a dep that truly is a conflict
        true_conflict_id = None
        for dep_id, info in medium_env._ground_truth.items():
            if info["is_conflict"]:
                true_conflict_id = dep_id
                break
        assert true_conflict_id, "Scenario must have at least one conflict"

        action = Action(
            action_type=ActionType.FLAG_CONFLICT,
            target_id=true_conflict_id,
            classification="GPL-2.0-only",
            category=LicenseCategory.COPYLEFT_STRONG,
            severity=ConflictSeverity.HIGH,
            confidence=0.9,
            reason="GPL-2.0 conflicts with proprietary project",
        )
        _, reward, _, _ = medium_env.step(action)
        assert reward.total > 0.0

    def test_flagging_false_positive_penalizes(self, medium_env):
        # Find a non-conflict dep
        safe_dep_id = None
        for dep_id, info in medium_env._ground_truth.items():
            if not info["is_conflict"]:
                safe_dep_id = dep_id
                break
        assert safe_dep_id, "Scenario must have at least one safe dep"

        action = Action(
            action_type=ActionType.FLAG_CONFLICT,
            target_id=safe_dep_id,
            classification="MIT",
            category=LicenseCategory.PERMISSIVE,
            severity=ConflictSeverity.HIGH,
            confidence=0.9,
        )
        _, reward, _, _ = medium_env.step(action)
        assert reward.total < 0.0, "False positive should penalize"

    def test_grader_f1_calculation(self, medium_env):
        # Perfect agent flags all real conflicts correctly
        for dep_id, info in medium_env._ground_truth.items():
            if info["is_conflict"]:
                action = Action(
                    action_type=ActionType.FLAG_CONFLICT,
                    target_id=dep_id,
                    classification=info["resolved_license"],
                    category=LicenseCategory.COPYLEFT_STRONG,
                    severity=ConflictSeverity(info["conflict_severity"]),
                    confidence=0.9,
                )
                medium_env.step(action)

        score = medium_env.final_score()
        assert score.total > 0.5, f"Perfect conflict detection should score > 0.5, got {score.total}"
        assert "f1" in score.breakdown
        assert score.breakdown["f1"] >= 0.9


# ──────────────────────────────────────────────────────────────────
# Task 3 (Hard) tests
# ──────────────────────────────────────────────────────────────────

class TestHardTask:
    def test_generate_report_ends_episode(self, hard_env):
        # Add a few findings first
        deps = hard_env._dependencies[:3]
        for dep in deps:
            action = Action(
                action_type=ActionType.ADD_FINDING,
                target_id=dep.dep_id,
                classification=dep.resolved_license or "MIT",
                category=LicenseCategory.PERMISSIVE,
                severity=ConflictSeverity.NONE,
                confidence=0.8,
            )
            hard_env.step(action)

        report = ComplianceReport(
            summary="Compliance scan complete. 3 critical findings identified.",
            total_dependencies=len(hard_env._dependencies),
            compliant_count=20,
            violation_count=4,
            unknown_count=2,
            recommended_actions=[
                "Replace mysqlclient with pymysql (MIT licensed)",
                "Replace ansible with fabric (BSD-2-Clause)",
                "Remove metabase-driver or self-host with source disclosure",
            ],
            overall_risk=ConflictSeverity.CRITICAL,
        )
        action = Action(
            action_type=ActionType.GENERATE_REPORT,
            report=report,
        )
        _, _, done, _ = hard_env.step(action)
        assert done is True

    def test_report_score_with_correct_overall_risk(self, hard_env):
        report = ComplianceReport(
            summary="Critical AGPL and GPL violations found in SaaS product dependencies.",
            total_dependencies=len(hard_env._dependencies),
            compliant_count=20,
            violation_count=4,
            unknown_count=2,
            recommended_actions=[
                "Remove metabase-driver (AGPL-3.0-only) — critical SaaS violation",
                "Remove grafana-client (AGPL-3.0-only) — critical SaaS violation",
                "Replace mysqlclient with pymysql",
                "Audit LGPL dynamic linking for paramiko and psycopg2",
            ],
            overall_risk=ConflictSeverity.CRITICAL,
        )
        action = Action(action_type=ActionType.GENERATE_REPORT, report=report)
        hard_env.step(action)
        score = hard_env.final_score()
        assert score.total >= 0.10, f"Reasonable report should score >= 0.10, got {score.total}"

    def test_missing_report_scores_zero_report_component(self, hard_env):
        score = hard_env.final_score()
        # report_score should be 0 when no report generated
        assert score.breakdown.get("report_score", 0.0) == 0.0

    def test_step_limit_terminates_episode(self, hard_env):
        hard_env._max_steps = 3
        action = Action(action_type=ActionType.MARK_REVIEWED, target_id="x")
        done = False
        for _ in range(3):
            _, _, done, _ = hard_env.step(action)
        assert done is True


# ──────────────────────────────────────────────────────────────────
# Reward model tests
# ──────────────────────────────────────────────────────────────────

class TestRewardModel:
    def test_reward_always_bounded(self):
        for task in ["classify_licenses", "detect_conflicts", "generate_compliance_report"]:
            env = LicenseComplianceEnv(task_id=task, seed=42)
            env.reset()
            score = env.final_score()
            assert 0.0 <= score.total <= 1.0, f"{task}: score out of bounds: {score.total}"

    def test_clarification_gives_no_reward(self, easy_env):
        action = Action(
            action_type=ActionType.REQUEST_CLARIFICATION,
            target_id="file_000",
        )
        _, reward, _, _ = easy_env.step(action)
        assert reward.total == 0.0

    def test_breakdown_keys_present(self, easy_env):
        for f in easy_env._files:
            action = Action(
                action_type=ActionType.CLASSIFY_LICENSE,
                target_id=f.file_id,
                classification="MIT",
                category=LicenseCategory.PERMISSIVE,
            )
            easy_env.step(action)
        score = easy_env.final_score()
        expected_keys = {"spdx_accuracy", "category_accuracy", "coverage"}
        assert expected_keys.issubset(set(score.breakdown.keys()))


# ──────────────────────────────────────────────────────────────────
# Brier calibration tests (new grader mechanic)
# ──────────────────────────────────────────────────────────────────

class TestBrierCalibration:
    def test_overconfident_wrong_penalised_more_than_uncertain_wrong(self):
        """Agent saying conf=0.99 and wrong should score lower than conf=0.5 and wrong."""
        from graders.grader_easy import score as easy_score
        from env.models import ScanFinding, LicenseCategory, ConflictSeverity

        gt = {
            "file_000": {"spdx_id": "MIT", "category": "permissive", "is_deceptive": False}
        }

        overconfident_wrong = [ScanFinding(
            finding_id="a", source_id="file_000",
            license_spdx="GPL-3.0-only",
            category=LicenseCategory.COPYLEFT_STRONG,
            severity=ConflictSeverity.NONE,
            agent_confidence=0.99,
        )]
        uncertain_wrong = [ScanFinding(
            finding_id="b", source_id="file_000",
            license_spdx="GPL-3.0-only",
            category=LicenseCategory.COPYLEFT_STRONG,
            severity=ConflictSeverity.NONE,
            agent_confidence=0.5,
        )]

        score_overconf = easy_score(overconfident_wrong, gt, 5, 20)
        score_uncertain = easy_score(uncertain_wrong, gt, 5, 20)

        assert score_overconf.total < score_uncertain.total, (
            f"Overconfident+wrong ({score_overconf.total}) should score lower "
            f"than uncertain+wrong ({score_uncertain.total})"
        )

    def test_confident_correct_scores_high_calibration(self):
        """Agent saying conf=0.95 and correct should have low Brier score."""
        from graders.grader_easy import score as easy_score
        from env.models import ScanFinding, LicenseCategory, ConflictSeverity

        gt = {
            "file_000": {"spdx_id": "MIT", "category": "permissive", "is_deceptive": False}
        }
        finding = [ScanFinding(
            finding_id="a", source_id="file_000",
            license_spdx="MIT",
            category=LicenseCategory.PERMISSIVE,
            severity=ConflictSeverity.NONE,
            agent_confidence=0.95,
        )]
        r = easy_score(finding, gt, 5, 20)
        assert r.breakdown["mean_brier_score"] < 0.01

    def test_brier_breakdown_key_present(self, easy_env):
        for f in easy_env._files:
            action = Action(
                action_type=ActionType.CLASSIFY_LICENSE,
                target_id=f.file_id,
                classification="MIT",
                category=LicenseCategory.PERMISSIVE,
                confidence=0.8,
            )
            easy_env.step(action)
        score = easy_env.final_score()
        assert "brier_calibration" in score.breakdown
        assert "mean_brier_score" in score.breakdown


# ──────────────────────────────────────────────────────────────────
# Deceptive license tests
# ──────────────────────────────────────────────────────────────────

class TestDeceptiveLicenses:
    def test_scenario_contains_deceptive_files(self):
        env = LicenseComplianceEnv(task_id="classify_licenses", seed=42)
        env.reset()
        deceptive = [
            fid for fid, info in env._ground_truth.items()
            if info.get("is_deceptive")
        ]
        assert len(deceptive) == 2, f"Expected 2 deceptive files, got {len(deceptive)}"

    def test_deceptive_file_has_misleading_header(self):
        env = LicenseComplianceEnv(task_id="classify_licenses", seed=42)
        env.reset()
        for fid, info in env._ground_truth.items():
            if info.get("is_deceptive"):
                file_entry = next(f for f in env._files if f.file_id == fid)
                # Header should mention the wrong license
                assert info["misleading_header"] in file_entry.content_snippet
                # Body SPDX should differ from the header
                assert info["spdx_id"] != info["misleading_header"]
                break

    def test_trusting_header_on_deceptive_file_scores_zero(self):
        from graders.grader_easy import score as easy_score
        from env.models import ScanFinding, LicenseCategory, ConflictSeverity

        env = LicenseComplianceEnv(task_id="classify_licenses", seed=42)
        env.reset()

        deceptive_id = next(
            fid for fid, info in env._ground_truth.items() if info.get("is_deceptive")
        )
        misleading = env._ground_truth[deceptive_id]["misleading_header"]
        true_spdx  = env._ground_truth[deceptive_id]["spdx_id"]

        # Agent trusts the header (wrong)
        header_follower = [ScanFinding(
            finding_id="x", source_id=deceptive_id,
            license_spdx=misleading,
            category=LicenseCategory.PERMISSIVE,
            severity=ConflictSeverity.NONE,
            agent_confidence=0.9,
        )]
        # Agent reads the body (correct)
        body_reader = [ScanFinding(
            finding_id="y", source_id=deceptive_id,
            license_spdx=true_spdx,
            category=LicenseCategory.COPYLEFT_STRONG,
            severity=ConflictSeverity.NONE,
            agent_confidence=0.9,
        )]

        gt = {deceptive_id: env._ground_truth[deceptive_id]}
        s_header = easy_score(header_follower, gt, 5, 20)
        s_body   = easy_score(body_reader, gt, 5, 20)

        assert s_body.total > s_header.total
        assert s_body.breakdown["deceptive_bonus"] > 0.0
        assert s_header.breakdown["deceptive_bonus"] == 0.0

    def test_deceptive_bonus_in_breakdown(self, easy_env):
        """Final score breakdown always includes deceptive_bonus key."""
        for f in easy_env._files:
            truth = easy_env._ground_truth.get(f.file_id, {})
            action = Action(
                action_type=ActionType.CLASSIFY_LICENSE,
                target_id=f.file_id,
                classification=truth.get("spdx_id", "MIT"),
                category=LicenseCategory(truth.get("category", "permissive")),
                confidence=0.9,
            )
            easy_env.step(action)
        score = easy_env.final_score()
        assert "deceptive_bonus" in score.breakdown


# ──────────────────────────────────────────────────────────────────
# Hard task: intentional ceiling documentation test
# ──────────────────────────────────────────────────────────────────

class TestHardTaskCeiling:
    def test_guessing_spdx_for_unknown_files_scores_zero(self):
        """Unknown vendor files must be flagged as 'unknown' category to earn credit."""
        from graders.grader_hard import score as hard_score
        from env.models import ScanFinding, LicenseCategory, ConflictSeverity

        env = LicenseComplianceEnv(task_id="generate_compliance_report", seed=42)
        env.reset()
        gt = env._ground_truth
        unknown_ids = gt.get("expected_unknown_files", [])
        assert unknown_ids, "Hard scenario must have unknown files"

        # Agent guesses MIT for unknown files (wrong — should not earn credit)
        guess_findings = [
            ScanFinding(
                finding_id=f"g{i}", source_id=fid,
                license_spdx="MIT", category=LicenseCategory.PERMISSIVE,
                severity=ConflictSeverity.NONE, agent_confidence=0.6,
            )
            for i, fid in enumerate(unknown_ids)
        ]
        r_guess = hard_score(guess_findings, None, gt, 10, 50)

        # Agent correctly flags unknown files as unknown category
        flag_findings = [
            ScanFinding(
                finding_id=f"f{i}", source_id=fid,
                license_spdx="UNKNOWN", category=LicenseCategory.UNKNOWN,
                severity=ConflictSeverity.NONE, agent_confidence=0.3,
            )
            for i, fid in enumerate(unknown_ids)
        ]
        r_flag = hard_score(flag_findings, None, gt, 10, 50)

        assert r_flag.breakdown["unknown_handling"] > 0.0
        assert r_guess.breakdown["unknown_handling"] == 0.0

    def test_oracle_ceiling_below_one(self):
        """Hard task oracle ceiling intentionally caps around 0.75–0.80, not 1.0."""
        env = LicenseComplianceEnv(task_id="generate_compliance_report", seed=42)
        env.reset()
        gt = env._ground_truth

        # Perfect oracle: flags all, correct categories, correct remediations
        from env.models import ScanFinding, LicenseCategory, ConflictSeverity, ComplianceReport
        from graders.grader_hard import score as hard_score

        findings = []
        for dep in env._dependencies:
            info = gt["deps"][dep.dep_id]
            sev = ConflictSeverity(info["severity"])
            findings.append(ScanFinding(
                finding_id=dep.dep_id[:8], source_id=dep.dep_id,
                license_spdx=info["resolved_license"],
                category=LicenseCategory.COPYLEFT_NETWORK if "AGPL" in info["resolved_license"] else LicenseCategory.COPYLEFT_STRONG,
                severity=sev,
                agent_confidence=1.0,
                remediation="Replace with permissive-licensed alternative" if info["severity"] == "critical" else None,
            ))
        # Flag unknown files correctly
        for f in env._files:
            if f.file_id in gt.get("expected_unknown_files", []):
                findings.append(ScanFinding(
                    finding_id=f.file_id[:8], source_id=f.file_id,
                    license_spdx="UNKNOWN", category=LicenseCategory.UNKNOWN,
                    severity=ConflictSeverity.NONE, agent_confidence=0.3,
                ))

        report = ComplianceReport(
            summary="Critical AGPL-3.0 and GPL violations found in SaaS project. Immediate remediation required.",
            total_dependencies=len(env._dependencies),
            compliant_count=20,
            violation_count=gt["total_violations"],
            unknown_count=len(gt["expected_unknown_files"]),
            recommended_actions=[
                "Remove metabase-driver (AGPL-3.0)",
                "Remove grafana-client (AGPL-3.0)",
                "Replace mysqlclient with pymysql",
            ],
            overall_risk=ConflictSeverity.CRITICAL,
        )

        r = hard_score(findings, report, gt, 48, 50)
        # Oracle should score well but not 1.0 — ceiling is intentional
        assert r.total >= 0.70, f"Oracle should score >= 0.70, got {r.total}"
        assert r.total <= 0.90, f"Oracle should not reach 1.0 (ceiling is intentional), got {r.total}"