"""
LicenseComplianceEnv — main OpenEnv-compliant environment class.

Implements the standard OpenEnv interface:
  reset() -> Observation
  step(action) -> (Observation, Reward, done, info)
  state() -> EnvironmentState
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from env.models import (
    Action, ActionType, Observation, Reward, EnvironmentState,
    FileEntry, DependencyEntry, ScanFinding, ComplianceReport,
    LicenseCategory, ConflictSeverity,
)
from data.license_corpus import LICENSE_CORPUS, get_conflict_severity


TaskId = Literal["classify_licenses", "detect_conflicts", "generate_compliance_report"]


class LicenseComplianceEnv:
    """
    OpenEnv-compliant environment for license compliance scanning tasks.
    
    Usage:
        env = LicenseComplianceEnv(task_id="classify_licenses", seed=42)
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        state = env.state()
    """

    def __init__(self, task_id: TaskId = "classify_licenses", seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self._step_count = 0
        self._max_steps = 20
        self._done = False
        self._cumulative_reward = 0.0
        self._findings: list[ScanFinding] = []
        self._report: ComplianceReport | None = None
        self._scenario: dict[str, Any] = {}
        self._files: list[FileEntry] = []
        self._dependencies: list[DependencyEntry] = []
        self._policy = None
        self._ground_truth: dict[str, Any] = {}
        self._messages: list[str] = []

    # ──────────────────────────────────────────────────────────────
    # OpenEnv API
    # ──────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._findings = []
        self._report = None
        self._messages = []

        scenario = self._load_scenario()
        self._scenario = scenario
        self._files = scenario["files"]
        self._dependencies = scenario["dependencies"]
        self._policy = scenario["policy"]
        self._ground_truth = scenario["ground_truth"]
        self._max_steps = scenario["max_steps"]

        self._messages.append(
            f"[ENV] Task '{self.task_id}' initialized. "
            f"Files: {len(self._files)}, Deps: {len(self._dependencies)}. "
            f"Max steps: {self._max_steps}."
        )

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """Execute one agent action and return (obs, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        reward = self._process_action(action)
        self._cumulative_reward = min(1.0, self._cumulative_reward + reward.total * 0.1)

        # Check terminal conditions
        done = self._check_done(action)
        self._done = done

        obs = self._build_observation()
        obs.done = done

        info = {
            "step": self._step_count,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "reward_breakdown": reward.breakdown,
        }

        return obs, reward, done, info

    def state(self) -> EnvironmentState:
        """Return full internal state (useful for debugging and oracles)."""
        return EnvironmentState(
            task_id=self.task_id,
            step=self._step_count,
            max_steps=self._max_steps,
            files=self._files,
            dependencies=self._dependencies,
            policy=self._policy,
            findings=self._findings,
            ground_truth=self._ground_truth,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
        )

    def final_score(self) -> Reward:
        """Compute final episode score using the task-specific grader."""
        return self._run_grader()

    @property
    def step_count(self) -> int:
        return self._step_count

    @step_count.setter
    def step_count(self, value: int) -> None:
        self._step_count = value

    @property
    def done(self) -> bool:
        return self._done

    @done.setter
    def done(self, value: bool) -> None:
        self._done = value

    @property
    def findings(self) -> list[ScanFinding]:
        return self._findings

    @findings.setter
    def findings(self, value: list[ScanFinding]) -> None:
        self._findings = list(value)

    @property
    def ground_truth(self) -> dict[str, Any]:
        return self._ground_truth

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    @cumulative_reward.setter
    def cumulative_reward(self, value: float) -> None:
        self._cumulative_reward = value

    @property
    def report(self) -> ComplianceReport | None:
        return self._report

    @report.setter
    def report(self, value: ComplianceReport | None) -> None:
        self._report = value

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _load_scenario(self) -> dict[str, Any]:
        if self.task_id == "classify_licenses":
            from tasks.task_easy import build_scenario
        elif self.task_id == "detect_conflicts":
            from tasks.task_medium import build_scenario
        elif self.task_id == "generate_compliance_report":
            from tasks.task_hard import build_scenario
        else:
            raise ValueError(f"Unknown task_id: {self.task_id}")
        return build_scenario(seed=self.seed)

    def _build_observation(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            step=self._step_count,
            max_steps=self._max_steps,
            files_to_scan=self._files,
            dependencies=self._dependencies,
            policy=self._policy,
            scan_results=list(self._findings),
            messages=list(self._messages[-5:]),  # last 5 messages
            done=self._done,
        )

    def _process_action(self, action: Action) -> Reward:
        """Process an action and return a step-level reward signal."""
        at = action.action_type

        if at == ActionType.CLASSIFY_LICENSE:
            return self._handle_classify(action)
        elif at == ActionType.FLAG_CONFLICT:
            return self._handle_flag_conflict(action)
        elif at == ActionType.ADD_FINDING:
            return self._handle_add_finding(action)
        elif at == ActionType.PROPOSE_REMEDIATION:
            return self._handle_remediation(action)
        elif at == ActionType.GENERATE_REPORT:
            return self._handle_generate_report(action)
        elif at == ActionType.MARK_REVIEWED:
            return self._handle_mark_reviewed(action)
        elif at == ActionType.REQUEST_CLARIFICATION:
            # No reward but valid action — agent gets a hint
            self._messages.append(
                f"[ENV] Clarification for '{action.target_id}': "
                "Check SPDX identifiers at https://spdx.org/licenses/"
            )
            return Reward(total=0.0, message="Clarification provided, no reward.")
        else:
            return Reward(total=0.0, message=f"Unknown action type: {at}")

    def _handle_classify(self, action: Action) -> Reward:
        file_id = action.target_id
        truth = (self._ground_truth if self.task_id == "classify_licenses"
                 else {}).get(file_id)

        finding = ScanFinding(
            finding_id=str(uuid.uuid4())[:8],
            source_id=file_id or "unknown",
            license_spdx=action.classification or "UNKNOWN",
            category=action.category or LicenseCategory.UNKNOWN,
            severity=ConflictSeverity.NONE,
            agent_confidence=action.confidence,
        )

        # Replace existing finding for same source
        self._findings = [f for f in self._findings if f.source_id != file_id]
        self._findings.append(finding)

        # Step-level partial reward
        if truth:
            spdx_ok = (action.classification or "").strip() == truth.get("spdx_id", "")
            cat_ok = (action.category and action.category.value == truth.get("category"))
            step_reward = (0.1 if spdx_ok else 0.0) + (0.04 if cat_ok else 0.0)
            msg = f"{'✓' if spdx_ok else '✗'} {file_id}: classified as {action.classification}"
        else:
            step_reward = 0.01  # no ground truth for this file
            msg = f"Classified {file_id} (no GT available)"

        self._messages.append(f"[CLASSIFY] {msg}")
        return Reward(total=min(1.0, step_reward), message=msg)

    def _handle_flag_conflict(self, action: Action) -> Reward:
        dep_id = action.target_id
        truth_deps = (
            self._ground_truth if self.task_id == "detect_conflicts"
            else self._ground_truth.get("deps", {})
        )
        truth = truth_deps.get(dep_id)

        finding = ScanFinding(
            finding_id=str(uuid.uuid4())[:8],
            source_id=dep_id or "unknown",
            license_spdx=action.classification or "UNKNOWN",
            category=action.category or LicenseCategory.UNKNOWN,
            severity=action.severity or ConflictSeverity.HIGH,
            conflict_reason=action.reason,
            remediation=action.remediation,
            agent_confidence=action.confidence,
        )
        self._findings = [f for f in self._findings if f.source_id != dep_id]
        self._findings.append(finding)

        if truth:
            is_real_conflict = truth.get("is_conflict", False)
            true_sev = truth.get("conflict_severity", "none")
            agent_sev = (action.severity or ConflictSeverity.HIGH).value

            if is_real_conflict:
                sev_diff = abs(
                    ["none","low","medium","high","critical"].index(agent_sev)
                    - ["none","low","medium","high","critical"].index(true_sev)
                )
                step_reward = 0.15 - sev_diff * 0.04
                msg = f"✓ Conflict correctly flagged: {dep_id} (sev={agent_sev})"
            else:
                step_reward = -0.10  # false positive
                msg = f"✗ False positive: {dep_id} is not a conflict"
        else:
            step_reward = 0.05
            msg = f"Flagged {dep_id} (no GT)"

        self._messages.append(f"[CONFLICT] {msg}")
        return Reward(total=max(-1.0, step_reward), message=msg)

    def _handle_add_finding(self, action: Action) -> Reward:
        """Generic finding addition for hard task."""
        finding = ScanFinding(
            finding_id=str(uuid.uuid4())[:8],
            source_id=action.target_id or "unknown",
            license_spdx=action.classification or "UNKNOWN",
            category=action.category or LicenseCategory.UNKNOWN,
            severity=action.severity or ConflictSeverity.MEDIUM,
            conflict_reason=action.reason,
            remediation=action.remediation,
            agent_confidence=action.confidence,
        )
        self._findings = [f for f in self._findings if f.source_id != action.target_id]
        self._findings.append(finding)
        self._messages.append(f"[FINDING] Added finding for {action.target_id}")
        return Reward(total=0.02, message="Finding recorded.")

    def _handle_remediation(self, action: Action) -> Reward:
        """Attach remediation to an existing finding."""
        for f in self._findings:
            if f.source_id == action.target_id:
                f.remediation = action.remediation
                f.reviewed = True
                self._messages.append(f"[REMEDIATE] Remediation proposed for {action.target_id}")
                return Reward(total=0.05, message="Remediation attached.")
        return Reward(total=0.0, message=f"No finding for {action.target_id}")

    def _handle_mark_reviewed(self, action: Action) -> Reward:
        for f in self._findings:
            if f.source_id == action.target_id:
                f.reviewed = True
        return Reward(total=0.01, message=f"Marked {action.target_id} as reviewed.")

    def _handle_generate_report(self, action: Action) -> Reward:
        if action.report:
            self._report = action.report
            self._done = True
            final = self._run_grader()
            self._messages.append(f"[REPORT] Final score: {final.total:.3f}")
            return final
        return Reward(total=0.0, message="Report action missing report field.")

    def _check_done(self, action: Action) -> bool:
        if self._step_count >= self._max_steps:
            return True
        if action.action_type == ActionType.GENERATE_REPORT and action.report:
            return True
        # For easy task: done when all files classified
        if self.task_id == "classify_licenses":
            classified_ids = {f.source_id for f in self._findings}
            all_ids = {f.file_id for f in self._files}
            if all_ids and all_ids.issubset(classified_ids):
                return True
        return False

    def _run_grader(self) -> Reward:
        if self.task_id == "classify_licenses":
            from graders.grader_easy import score
            return score(self._findings, self._ground_truth, self._step_count, self._max_steps)
        elif self.task_id == "detect_conflicts":
            from graders.grader_medium import score
            return score(self._findings, self._ground_truth, self._step_count, self._max_steps)
        elif self.task_id == "generate_compliance_report":
            from graders.grader_hard import score
            return score(self._findings, self._report, self._ground_truth, self._step_count, self._max_steps)
        raise ValueError(f"No grader for task: {self.task_id}")
