"""
Task 3 (Hard): Full Compliance Report Generation
Agent must triage a mixed SaaS repo, resolve unknowns, and produce
a structured ComplianceReport. Graded on completeness + accuracy + actionability.
"""
from __future__ import annotations
import random
from typing import Any

from env.models import (
    FileEntry, DependencyEntry, PolicyConfig,
    LicenseCategory, ScanFinding, ConflictSeverity
)
from data.license_corpus import LICENSE_CORPUS, get_conflict_severity


SAAS_DEPS = [
    ("fastapi",         "0.110.0", "MIT"),
    ("uvicorn",         "0.27.0",  "BSD-3-Clause"),
    ("sqlalchemy",      "2.0.28",  "MIT"),
    ("alembic",         "1.13.0",  "MIT"),
    ("celery",          "5.3.6",   "BSD-3-Clause"),
    ("redis",           "5.0.1",   "MIT"),
    ("boto3",           "1.34.0",  "Apache-2.0"),
    ("stripe",          "7.9.0",   "Apache-2.0"),
    ("pyjwt",           "2.8.0",   "MIT"),
    ("pillow",          "10.2.0",  "HPND"),
    ("ffmpeg-python",   "0.2.0",   "Apache-2.0"),
    ("mysqlclient",     "2.2.0",   "GPL-2.0-only"),  # CONFLICT
    ("ansible",         "8.5.0",   "GPL-3.0-only"),  # CONFLICT
    ("minio",           "7.2.0",   "Apache-2.0"),
    ("elasticsearch",   "8.10.0",  "Apache-2.0"),
    ("paramiko",        "3.4.0",   "LGPL-2.1-only"),
    ("psycopg2",        "2.9.9",   "LGPL-2.1-only"),
    ("pydantic",        "2.6.0",   "MIT"),
    ("httpx",           "0.26.0",  "BSD-3-Clause"),
    ("sentry-sdk",      "1.39.0",  "MIT"),
    ("opentelemetry",   "1.22.0",  "Apache-2.0"),
    ("posthog",         "3.1.0",   "MIT"),
    ("dbt-core",        "1.7.0",   "Apache-2.0"),
    ("metabase-driver", "0.47.0",  "AGPL-3.0-only"),  # CRITICAL for SaaS
    ("grafana-client",  "3.5.0",   "AGPL-3.0-only"),  # CRITICAL for SaaS
    ("networkx",        "3.2.1",   "BSD-3-Clause"),
]


VENDOR_FILES = [
    ("vendor/libmagic/LICENSE",      "MIT",
     "MIT License\nPermission is hereby granted, free of charge..."),
    ("vendor/libssl/COPYING",        "OpenSSL",
     "The OpenSSL toolkit stays under a dual license, i.e. both the conditions of the OpenSSL License and the original SSLeay license apply to the toolkit."),
    ("vendor/libjpeg/README",        "IJG",
     "The authors make NO WARRANTY or representation, either express or implied... This software is provided 'as-is', without any express or implied warranty."),
    ("NOTICE",                       "Apache-2.0",
     "This product includes software developed by the Apache Software Foundation (http://www.apache.org/)."),
    ("vendor/ffmpeg/LICENSE.md",     "GPL-2.0-only",
     "FFmpeg is licensed under the GNU General Public License version 2 or later."),
    ("frontend/node_modules/.package-lock.json", "UNKNOWN",
     "# mixed licenses - see individual package.json files"),
]


def build_scenario(seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)

    deps = [
        DependencyEntry(
            dep_id=f"dep_{i:03d}",
            name=name, version=ver, declared_license=lic,
            resolved_license=lic,
            is_direct=rng.random() > 0.3,
            used_in=rng.choice(["runtime", "runtime", "dev"]),
        )
        for i, (name, ver, lic) in enumerate(SAAS_DEPS)
    ]

    files = [
        FileEntry(
            file_id=f"file_{i:03d}",
            path=path, content_snippet=snippet,
            suspected_license=lic,
        )
        for i, (path, lic, snippet) in enumerate(VENDOR_FILES)
    ]

    policy = PolicyConfig(
        project_license="PROPRIETARY",
        allowed_categories=[
            LicenseCategory.PERMISSIVE,
            LicenseCategory.PUBLIC_DOMAIN,
            LicenseCategory.COPYLEFT_WEAK,   # LGPL OK if dynamically linked
        ],
        forbidden_licenses=["GPL-2.0-only", "GPL-3.0-only", "AGPL-3.0-only"],
        copyleft_allowed_in_dev=True,
        network_copyleft_forbidden=True,
        require_attribution=True,
        distribution_type="saas",
    )

    # Build ground truth
    ground_truth_deps: dict[str, dict] = {}
    for dep in deps:
        sev = get_conflict_severity("PROPRIETARY", dep.resolved_license or "", "saas")
        ground_truth_deps[dep.dep_id] = {
            "resolved_license": dep.resolved_license,
            "severity": sev,
            "is_conflict": sev != "none",
        }

    expected_critical = [
        dep.dep_id for dep in deps
        if ground_truth_deps[dep.dep_id]["severity"] == "critical"
    ]
    expected_high = [
        dep.dep_id for dep in deps
        if ground_truth_deps[dep.dep_id]["severity"] == "high"
    ]
    expected_unknown_files = [
        f.file_id for f in files
        if f.suspected_license in ("UNKNOWN", "OpenSSL", "IJG")
    ]

    return {
        "task_id": "generate_compliance_report",
        "files": files,
        "dependencies": deps,
        "policy": policy,
        "ground_truth": {
            "deps": ground_truth_deps,
            "expected_critical": expected_critical,
            "expected_high": expected_high,
            "expected_unknown_files": expected_unknown_files,
            "total_violations": len(expected_critical) + len(expected_high),
            "must_remediate": expected_critical,
        },
        "max_steps": 50,
    }