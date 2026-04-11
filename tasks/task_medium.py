"""
Task 2 (Medium): Dependency Conflict Detection
Agent must find all license conflicts in a dependency manifest
given the project's policy constraints.
"""
from __future__ import annotations
import random
from typing import Any

from env.models import DependencyEntry, PolicyConfig, LicenseCategory
from data.license_corpus import LICENSE_CORPUS, CONFLICT_RULES, get_conflict_severity


PACKAGE_NAMES = [
    ("requests", "2.31.0", "Apache-2.0"),
    ("numpy", "1.26.0", "BSD-3-Clause"),
    ("flask", "3.0.0", "BSD-3-Clause"),
    ("sqlalchemy", "2.0.0", "MIT"),
    ("pydantic", "2.5.0", "MIT"),
    ("celery", "5.3.0", "BSD-3-Clause"),
    ("redis-py", "5.0.0", "MIT"),
    ("aiohttp", "3.9.0", "Apache-2.0"),
    ("pillow", "10.1.0", "HPND"),         # Uncommon - maps to permissive
    ("paramiko", "3.3.0", "LGPL-2.1-only"),
    ("cryptography", "41.0.0", "Apache-2.0"),
    ("pymysql", "1.1.0", "MIT"),
    ("ffmpeg-python", "0.2.0", "Apache-2.0"),
    ("mysqlclient", "2.2.0", "GPL-2.0-only"),
    ("psutil", "5.9.0", "BSD-3-Clause"),
    ("ansible", "8.5.0", "GPL-3.0-only"),
    ("netaddr", "0.9.0", "BSD-3-Clause"),
    ("charset-normalizer", "3.3.0", "MIT"),
    ("grpcio", "1.59.0", "Apache-2.0"),
    ("elasticsearch-py", "8.10.0", "Apache-2.0"),
]


def build_scenario(seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    project_license = "PROPRIETARY"

    pkgs = list(PACKAGE_NAMES)
    rng.shuffle(pkgs)
    selected = pkgs[:12]

    dependencies: list[DependencyEntry] = []
    ground_truth: dict[str, dict] = {}

    for i, (name, version, lic) in enumerate(selected):
        dep_id = f"dep_{i:03d}"
        # Occasionally make declared license slightly ambiguous
        declared = lic
        if rng.random() > 0.8 and lic in LICENSE_CORPUS:
            aliases = LICENSE_CORPUS[lic].get("aliases", [])
            if aliases:
                declared = rng.choice(aliases)

        dep = DependencyEntry(
            dep_id=dep_id,
            name=name,
            version=version,
            declared_license=declared,
            resolved_license=lic,
            is_direct=rng.random() > 0.4,
            used_in=rng.choice(["runtime", "dev", "runtime", "runtime"]),
        )
        dependencies.append(dep)

        severity = get_conflict_severity(project_license, lic, "proprietary")
        ground_truth[dep_id] = {
            "resolved_license": lic,
            "conflict_severity": severity,
            "is_conflict": severity != "none",
        }

    policy = PolicyConfig(
        project_license=project_license,
        allowed_categories=[LicenseCategory.PERMISSIVE, LicenseCategory.PUBLIC_DOMAIN],
        forbidden_licenses=["GPL-2.0-only", "GPL-3.0-only", "AGPL-3.0-only"],
        copyleft_allowed_in_dev=True,
        network_copyleft_forbidden=True,
        distribution_type="proprietary",
    )

    return {
        "task_id": "detect_conflicts",
        "files": [],
        "dependencies": dependencies,
        "policy": policy,
        "ground_truth": ground_truth,
        "max_steps": 30,
    }