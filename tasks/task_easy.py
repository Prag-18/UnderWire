"""
Task 1 (Easy): License Classification

Agent receives files containing raw license text snippets and must identify
the canonical SPDX identifier + category for each.

Deceptive files (injected for ~2 of 8 slots)
─────────────────────────────────────────────
These files have a misleading SPDX-License-Identifier header (e.g. "MIT")
while the body text is actually from a different license (e.g. GPL-3.0).
This tests whether the agent reads the full body rather than trusting headers.
Ground truth marks these with "is_deceptive": True.
The correct answer is the license the BODY text belongs to — not the header.
Reward: grader_easy gives a +0.03 bonus per correctly resolved deceptive file.
"""
from __future__ import annotations
import random
from typing import Any

from env.models import FileEntry, PolicyConfig, LicenseCategory
from data.license_corpus import LICENSE_CORPUS


# Deceptive pairs: (header_spdx, body_spdx)
# Header claims one license; body text is from another.
DECEPTIVE_PAIRS = [
    ("MIT",         "GPL-3.0-only"),
    ("Apache-2.0",  "AGPL-3.0-only"),
    ("ISC",         "GPL-2.0-only"),
    ("BSD-3-Clause","LGPL-2.1-only"),
    ("MIT",         "MPL-2.0"),
]


def build_scenario(seed: int = 42) -> dict[str, Any]:
    """Generate a deterministic classification scenario with deceptive files."""
    rng = random.Random(seed)
    licenses_to_test = list(LICENSE_CORPUS.keys())
    rng.shuffle(licenses_to_test)

    # Reserve 2 slots for deceptive files, 6 for honest
    deceptive_count = 2
    honest_count    = 6
    honest_licenses = licenses_to_test[:honest_count]

    files: list[FileEntry] = []
    ground_truth: dict[str, dict] = {}

    # ── Honest files ──────────────────────────────────────────────
    for i, spdx_id in enumerate(honest_licenses):
        meta    = LICENSE_CORPUS[spdx_id]
        snippet = meta["snippet"]

        if rng.random() > 0.5:
            alias   = rng.choice(meta["aliases"]) if meta["aliases"] else spdx_id
            snippet = f"// SPDX-License-Identifier: {alias}\n\n" + snippet[:250]

        file_id = f"file_{i:03d}"
        files.append(FileEntry(
            file_id=file_id,
            path=f"src/vendor/lib_{i}/LICENSE",
            content_snippet=snippet,
        ))
        ground_truth[file_id] = {
            "spdx_id":      spdx_id,
            "category":     meta["category"],
            "is_deceptive": False,
        }

    # ── Deceptive files ───────────────────────────────────────────
    deceptive_pool = list(DECEPTIVE_PAIRS)
    rng.shuffle(deceptive_pool)

    for j in range(deceptive_count):
        header_spdx, body_spdx = deceptive_pool[j % len(deceptive_pool)]
        body_meta  = LICENSE_CORPUS[body_spdx]

        # Fabricate a misleading file: wrong SPDX header, correct body text
        snippet = (
            f"// SPDX-License-Identifier: {header_spdx}\n"
            f"// Copyright (c) {2018 + j} Example Corp.\n\n"
            + body_meta["snippet"]
        )

        file_id = f"file_{honest_count + j:03d}"
        files.append(FileEntry(
            file_id=file_id,
            path=f"src/vendor/third_party_{j}/LICENSE.md",
            content_snippet=snippet,
            suspected_license=header_spdx,   # env signals the (wrong) header
        ))
        ground_truth[file_id] = {
            "spdx_id":      body_spdx,        # truth = body, not header
            "category":     body_meta["category"],
            "is_deceptive": True,
            "misleading_header": header_spdx,
        }

    rng.shuffle(files)   # randomise order so agent can't rely on position

    policy = PolicyConfig(
        project_license="MIT",
        allowed_categories=[LicenseCategory.PERMISSIVE, LicenseCategory.PUBLIC_DOMAIN],
        distribution_type="open-source",
    )

    return {
        "task_id":     "classify_licenses",
        "files":       files,
        "dependencies": [],
        "policy":      policy,
        "ground_truth": ground_truth,
        "max_steps":   20,
    }