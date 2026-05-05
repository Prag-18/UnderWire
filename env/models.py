"""
Typed Pydantic models for the License Compliance Scanner OpenEnv environment.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class LicenseCategory(str, Enum):
    PERMISSIVE = "permissive"
    COPYLEFT_WEAK = "copyleft_weak"
    COPYLEFT_STRONG = "copyleft_strong"
    COPYLEFT_NETWORK = "copyleft_network"
    PROPRIETARY = "proprietary"
    PUBLIC_DOMAIN = "public_domain"
    UNKNOWN = "unknown"


class ConflictSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class ActionType(str, Enum):
    CLASSIFY_LICENSE = "classify_license"
    FLAG_CONFLICT = "flag_conflict"
    MARK_REVIEWED = "mark_reviewed"
    REQUEST_CLARIFICATION = "request_clarification"
    ADD_FINDING = "add_finding"
    PROPOSE_REMEDIATION = "propose_remediation"
    GENERATE_REPORT = "generate_report"


class FileEntry(BaseModel):
    file_id: str
    path: str
    content_snippet: str
    suspected_license: Optional[str] = None
    reviewed: bool = False


class DependencyEntry(BaseModel):
    dep_id: str
    name: str
    version: str
    declared_license: Optional[str] = None
    resolved_license: Optional[str] = None
    is_direct: bool = True
    used_in: str = "runtime"


class PolicyConfig(BaseModel):
    project_license: str
    allowed_categories: list[LicenseCategory]
    forbidden_licenses: list[str] = Field(default_factory=list)
    copyleft_allowed_in_dev: bool = True
    network_copyleft_forbidden: bool = True
    require_attribution: bool = True
    distribution_type: str = "proprietary"


class ScanFinding(BaseModel):
    finding_id: str
    source_id: str
    license_spdx: str
    category: LicenseCategory
    severity: ConflictSeverity
    conflict_reason: Optional[str] = None
    remediation: Optional[str] = None
    agent_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    reviewed: bool = False
    # Phase 2 — AI Brain: Claude's reasoning text (None = rule-based result)
    ai_reasoning: Optional[str] = None


class ComplianceReport(BaseModel):
    summary: str
    total_dependencies: int
    compliant_count: int
    violation_count: int
    unknown_count: int
    critical_findings: list[ScanFinding] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    overall_risk: ConflictSeverity
    agent_notes: str = ""


class Observation(BaseModel):
    task_id: str
    step: int
    max_steps: int
    files_to_scan: list[FileEntry] = Field(default_factory=list)
    dependencies: list[DependencyEntry] = Field(default_factory=list)
    policy: Optional[PolicyConfig] = None
    scan_results: list[ScanFinding] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    action_type: ActionType
    target_id: Optional[str] = None
    classification: Optional[str] = None
    category: Optional[LicenseCategory] = None
    severity: Optional[ConflictSeverity] = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    reason: Optional[str] = None
    remediation: Optional[str] = None
    report: Optional[ComplianceReport] = None


class Reward(BaseModel):
    total: float = Field(ge=-1.0, le=1.0)
    breakdown: dict[str, float] = Field(default_factory=dict)
    message: str = ""


class EnvironmentState(BaseModel):
    task_id: str
    step: int
    max_steps: int
    files: list[FileEntry]
    dependencies: list[DependencyEntry]
    policy: Optional[PolicyConfig]
    findings: list[ScanFinding]
    ground_truth: dict[str, Any]
    cumulative_reward: float
    done: bool