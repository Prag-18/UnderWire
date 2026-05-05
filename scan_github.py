"""
scan_github.py — GitHub repository scanner for the UnderWire environment.

Given a GitHub repository URL (e.g. https://github.com/owner/repo), this
module:
  1. Converts it to a raw.githubusercontent.com URL for package.json or
     requirements.txt.
  2. Fetches the manifest via httpx.
  3. Parses it into a list of DependencyEntry objects with resolved licenses
     (using static NPM and PyPI license dicts, augmented by LICENSE_CORPUS).
  4. For packages not found in static maps, calls Claude AI to reason about
     the likely license (Phase 2 — AI Brain).
  5. Runs the detect_conflicts grader and returns structured findings JSON.

Supported manifest files (tried in order):
  - package.json   (Node/npm repos)
  - requirements.txt  (Python/PyPI repos)
"""
from __future__ import annotations

import os
import re
import json
import uuid as _uuid
from typing import Any

import httpx

from env.models import (
    DependencyEntry, ScanFinding, LicenseCategory,
    ConflictSeverity, PolicyConfig,
)
from data.license_corpus import LICENSE_CORPUS, get_conflict_severity

# ---------------------------------------------------------------------------
# Static license maps
# ---------------------------------------------------------------------------

NPM_LICENSE_MAP: dict[str, str] = {
    "react": "MIT",
    "react-dom": "MIT",
    "vue": "MIT",
    "angular": "MIT",
    "lodash": "MIT",
    "underscore": "MIT",
    "axios": "MIT",
    "express": "MIT",
    "moment": "MIT",
    "dayjs": "MIT",
    "chalk": "MIT",
    "commander": "MIT",
    "yargs": "MIT",
    "minimist": "MIT",
    "dotenv": "BSD-2-Clause",
    "debug": "MIT",
    "semver": "ISC",
    "glob": "ISC",
    "rimraf": "ISC",
    "mkdirp": "MIT",
    "uuid": "MIT",
    "async": "MIT",
    "bluebird": "MIT",
    "rxjs": "Apache-2.0",
    "typescript": "Apache-2.0",
    "webpack": "MIT",
    "babel-core": "MIT",
    "@babel/core": "MIT",
    "@babel/preset-env": "MIT",
    "eslint": "MIT",
    "prettier": "MIT",
    "jest": "MIT",
    "mocha": "MIT",
    "chai": "MIT",
    "sinon": "BSD-3-Clause",
    "nodemon": "MIT",
    "cors": "MIT",
    "body-parser": "MIT",
    "multer": "MIT",
    "mongoose": "MIT",
    "sequelize": "MIT",
    "knex": "MIT",
    "pg": "MIT",
    "mysql2": "MIT",
    "redis": "MIT",
    "socket.io": "MIT",
    "passport": "MIT",
    "jsonwebtoken": "MIT",
    "bcrypt": "MIT",
    "bcryptjs": "MIT",
    "helmet": "MIT",
    "morgan": "MIT",
    "winston": "MIT",
    "pino": "MIT",
    "sharp": "Apache-2.0",
    "jimp": "MIT",
    "node-fetch": "MIT",
    "cross-fetch": "MIT",
    "form-data": "MIT",
    "qs": "BSD-3-Clause",
    "query-string": "MIT",
    "classnames": "MIT",
    "immer": "MIT",
    "redux": "MIT",
    "mobx": "MIT",
    "zustand": "MIT",
    "next": "MIT",
    "nuxt": "MIT",
    "gatsby": "MIT",
    "vite": "MIT",
    "rollup": "MIT",
    "esbuild": "MIT",
    "tailwindcss": "MIT",
    "bootstrap": "MIT",
    "jquery": "MIT",
    "d3": "ISC",
    "chart.js": "MIT",
    "three": "MIT",
    "gsap": "GSAP-Standard-License",
    "electron": "MIT",
    "puppeteer": "Apache-2.0",
    "playwright": "Apache-2.0",
    "cypress": "MIT",
    "storybook": "MIT",
    "husky": "MIT",
    "lint-staged": "MIT",
    "concurrently": "MIT",
    "cross-env": "MIT",
    "dotenv-expand": "BSD-2-Clause",
    "xml2js": "MIT",
    "cheerio": "MIT",
    "marked": "MIT",
    "highlight.js": "BSD-3-Clause",
    "prismjs": "MIT",
    "dompurify": "Apache-2.0",
    "sanitize-html": "MIT",
    "zod": "MIT",
    "yup": "MIT",
    "formik": "Apache-2.0",
    "react-hook-form": "MIT",
    "swr": "MIT",
    "@tanstack/react-query": "MIT",
    "graphql": "MIT",
    "apollo-client": "MIT",
    "@apollo/client": "MIT",
    "mysql": "MIT",
    "sqlite3": "MIT",
    "better-sqlite3": "MIT",
    "typeorm": "MIT",
    "prisma": "Apache-2.0",
    "@prisma/client": "Apache-2.0",
    "stripe": "MIT",
    "twilio": "MIT",
    "nodemailer": "MIT",
    "aws-sdk": "Apache-2.0",
    "@aws-sdk/client-s3": "Apache-2.0",
    "firebase": "Apache-2.0",
    "supabase": "MIT",
    "@supabase/supabase-js": "MIT",
    "openai": "Apache-2.0",
    "langchain": "MIT",
    "handlebars": "MIT",
    "ejs": "Apache-2.0",
    "pug": "MIT",
    "nunjucks": "BSD-2-Clause",
    "node-gyp": "MIT",
    "node-addon-api": "MIT",
    "ffmpeg-static": "GPL-2.0-only",
    "node-ffmpeg": "GPL-2.0-only",
    "gpl-module": "GPL-3.0-only",
    "agpl-module": "AGPL-3.0-only",
}

PYPI_LICENSE_MAP: dict[str, str] = {
    "requests": "Apache-2.0",
    "urllib3": "MIT",
    "certifi": "MPL-2.0",
    "charset-normalizer": "MIT",
    "idna": "BSD-3-Clause",
    "numpy": "BSD-3-Clause",
    "pandas": "BSD-3-Clause",
    "matplotlib": "PSF-2.0",
    "scipy": "BSD-3-Clause",
    "scikit-learn": "BSD-3-Clause",
    "tensorflow": "Apache-2.0",
    "torch": "BSD-3-Clause",
    "torchvision": "BSD-3-Clause",
    "keras": "Apache-2.0",
    "flask": "BSD-3-Clause",
    "django": "BSD-3-Clause",
    "fastapi": "MIT",
    "uvicorn": "BSD-3-Clause",
    "starlette": "BSD-3-Clause",
    "pydantic": "MIT",
    "sqlalchemy": "MIT",
    "alembic": "MIT",
    "celery": "BSD-3-Clause",
    "redis": "MIT",
    "aiohttp": "Apache-2.0",
    "httpx": "BSD-3-Clause",
    "boto3": "Apache-2.0",
    "botocore": "Apache-2.0",
    "s3transfer": "Apache-2.0",
    "google-cloud-storage": "Apache-2.0",
    "google-auth": "Apache-2.0",
    "paramiko": "LGPL-2.1-only",
    "cryptography": "Apache-2.0",
    "pyopenssl": "Apache-2.0",
    "pillow": "HPND",
    "pil": "HPND",
    "opencv-python": "Apache-2.0",
    "imageio": "BSD-2-Clause",
    "pytest": "MIT",
    "pytest-asyncio": "Apache-2.0",
    "pytest-cov": "MIT",
    "coverage": "Apache-2.0",
    "tox": "MIT",
    "black": "MIT",
    "flake8": "MIT",
    "pylint": "GPL-2.0-only",
    "mypy": "MIT",
    "isort": "MIT",
    "pre-commit": "MIT",
    "setuptools": "MIT",
    "wheel": "MIT",
    "pip": "MIT",
    "virtualenv": "MIT",
    "poetry": "MIT",
    "click": "BSD-3-Clause",
    "rich": "MIT",
    "typer": "MIT",
    "colorama": "BSD-3-Clause",
    "tqdm": "MIT",
    "loguru": "MIT",
    "pyyaml": "MIT",
    "toml": "MIT",
    "tomli": "MIT",
    "dotenv": "BSD-3-Clause",
    "python-dotenv": "BSD-3-Clause",
    "jinja2": "BSD-3-Clause",
    "markupsafe": "BSD-3-Clause",
    "werkzeug": "BSD-3-Clause",
    "wtforms": "BSD-3-Clause",
    "marshmallow": "MIT",
    "attrs": "MIT",
    "cattrs": "MIT",
    "dacite": "MIT",
    "dataclasses-json": "MIT",
    "psycopg2": "LGPL-3.0-only",
    "psycopg2-binary": "LGPL-3.0-only",
    "psycopg": "LGPL-3.0-only",
    "pymysql": "MIT",
    "mysqlclient": "GPL-2.0-only",
    "pymongo": "Apache-2.0",
    "motor": "Apache-2.0",
    "elasticsearch": "Apache-2.0",
    "redis-py": "MIT",
    "kombu": "BSD-3-Clause",
    "billiard": "BSD-3-Clause",
    "amqp": "BSD-3-Clause",
    "vine": "BSD-3-Clause",
    "openai": "Apache-2.0",
    "langchain": "MIT",
    "anthropic": "MIT",
    "tiktoken": "MIT",
    "transformers": "Apache-2.0",
    "datasets": "Apache-2.0",
    "accelerate": "Apache-2.0",
    "diffusers": "Apache-2.0",
    "sentence-transformers": "Apache-2.0",
    "huggingface-hub": "Apache-2.0",
    "tokenizers": "Apache-2.0",
    "spacy": "MIT",
    "nltk": "Apache-2.0",
    "gensim": "LGPL-2.1-only",
    "textblob": "MIT",
    "ansible": "GPL-3.0-only",
    "netaddr": "BSD-3-Clause",
    "psutil": "BSD-3-Clause",
    "grpcio": "Apache-2.0",
    "protobuf": "BSD-3-Clause",
    "pika": "BSD-3-Clause",
    "ffmpeg-python": "Apache-2.0",
    "moviepy": "MIT",
    "librosa": "ISC",
    "soundfile": "BSD-3-Clause",
    "pyaudio": "MIT",
    "stripe": "MIT",
    "twilio": "MIT",
    "sendgrid": "MIT",
}

EXTRA_LICENSE_CATEGORIES: dict[str, str] = {
    "HPND":              "permissive",
    "PSF-2.0":           "permissive",
    "BSD-2-Clause":      "permissive",
    "LGPL-3.0-only":     "copyleft_weak",
    "LGPL-3.0-or-later": "copyleft_weak",
    "LGPL-2.1-or-later": "copyleft_weak",
    "EUPL-1.2":          "copyleft_weak",
    "CDDL-1.0":          "copyleft_weak",
    "EPL-2.0":           "copyleft_weak",
    "CPL-1.0":           "copyleft_weak",
    "GSAP-Standard-License": "proprietary",
    "CC-BY-4.0":         "permissive",
    "CC-BY-SA-4.0":      "copyleft_weak",
    "CC-BY-NC-4.0":      "proprietary",
    "SSPL-1.0":          "copyleft_network",
    "BSL-1.1":           "proprietary",
    "Elastic-2.0":       "proprietary",
}


def _resolve_license_category(spdx: str) -> str:
    if spdx in LICENSE_CORPUS:
        return LICENSE_CORPUS[spdx]["category"]
    return EXTRA_LICENSE_CATEGORIES.get(spdx, "unknown")


# ---------------------------------------------------------------------------
# Claude AI fallback for the scanner (Phase 2 — AI Brain)
# ---------------------------------------------------------------------------

def _claude_classify_for_scan(
    package_name: str,
    ecosystem: str,  # "npm" or "python"
) -> dict[str, str]:
    """
    Ask Claude to reason about the most likely SPDX license for a package
    that isn’t in the static maps.

    Returns:
      spdx_id   — SPDX identifier or "UNKNOWN"
      category  — license category string
      reasoning — Claude’s explanation (displayed in purple AI Analysis box)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"spdx_id": "UNKNOWN", "category": "unknown",
                "reasoning": "ANTHROPIC_API_KEY not configured."}

    try:
        import anthropic  # type: ignore
    except ImportError:
        return {"spdx_id": "UNKNOWN", "category": "unknown",
                "reasoning": "anthropic package not installed."}

    prompt = (
        "You are an expert open-source license compliance lawyer.\n"
        f"A {ecosystem} package named '{package_name}' has an unknown license.\n"
        "Based on your knowledge of this package, respond with ONLY a JSON object "
        "containing exactly three keys:\n"
        "  spdx_id   — the most likely SPDX identifier (e.g. MIT, Apache-2.0, "
        "GPL-3.0-only, UNKNOWN if truly unclear)\n"
        "  category  — one of: permissive, copyleft, copyleft_weak, "
        "copyleft_network, public_domain, proprietary, unknown\n"
        "  reasoning — one concise paragraph explaining your conclusion\n\n"
        "Respond with valid JSON only — no markdown, no preamble."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()
        result = json.loads(raw)
        return {
            "spdx_id": str(result.get("spdx_id", "UNKNOWN")),
            "category": str(result.get("category", "unknown")),
            "reasoning": str(result.get("reasoning", "")),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "spdx_id": "UNKNOWN",
            "category": "unknown",
            "reasoning": f"AI analysis failed: {exc}",
        }


def _github_url_to_raw(github_url: str, path: str, branch: str = "main") -> str:
    match = re.match(
        r"https?://github\.com/([^/]+)/([^/?\s#]+)", github_url.strip()
    )
    if not match:
        raise ValueError(f"Not a valid GitHub URL: {github_url!r}")
    owner, repo = match.group(1), match.group(2).rstrip("/")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def _fetch_raw(url: str, timeout: float = 10.0) -> str | None:
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise


# ---------------------------------------------------------------------------
# Package.json parser
# ---------------------------------------------------------------------------

def _normalize_version(ver: str) -> str:
    return re.sub(r"^[\^~>=<!\s]+", "", ver).split(" ")[0].strip() or ver


def parse_package_json(content: str, project_license: str = "PROPRIETARY") -> list[DependencyEntry]:
    """
    Parse a package.json manifest into DependencyEntry objects.

    Reads both ``dependencies`` and ``devDependencies``. Maps each package
    name to an SPDX license using NPM_LICENSE_MAP. Falls back to None for
    unknown packages.
    """
    try:
        pkg = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid package.json: {exc}") from exc

    entries: list[DependencyEntry] = []
    idx = 0

    deps: dict[str, str] = dict(pkg.get("dependencies", {}))
    dev_deps: dict[str, str] = pkg.get("devDependencies", {})

    def _add(name: str, version_raw: str, is_dev: bool) -> None:
        nonlocal idx
        version = _normalize_version(version_raw)
        lower = name.lower()
        spdx = NPM_LICENSE_MAP.get(name) or NPM_LICENSE_MAP.get(lower)
        dep_id = f"npm_{idx:03d}"
        entries.append(DependencyEntry(
            dep_id=dep_id,
            name=name,
            version=version,
            declared_license=spdx,
            resolved_license=spdx,
            is_direct=True,
            used_in="dev" if is_dev else "runtime",
        ))
        idx += 1

    for name, ver in deps.items():
        _add(name, ver, is_dev=False)
    for name, ver in dev_deps.items():
        _add(name, ver, is_dev=True)

    return entries


# ---------------------------------------------------------------------------
# requirements.txt parser
# ---------------------------------------------------------------------------

_REQ_LINE = re.compile(
    r"^([A-Za-z0-9_.\-]+)"
    r"(?:\[.*?\])?"
    r"\s*"
    r"((?:[><=!~]{1,2}[^\s,;]+)"
    r"(?:\s*,\s*(?:[><=!~]{1,2}[^\s,;]+))*)?"
)


def _parse_version_spec(spec: str) -> str:
    if not spec:
        return "*"
    for op in ("==", "~=", ">=", "<=", ">", "<"):
        match = re.search(re.escape(op) + r"([^\s,;]+)", spec)
        if match:
            return match.group(1).strip()
    return spec.strip() or "*"


def parse_requirements_txt(content: str, project_license: str = "PROPRIETARY") -> list[DependencyEntry]:
    """
    Parse a requirements.txt into DependencyEntry objects.

    Handles ==, >=, ~=, <=, >, < specifiers, inline comments, -r directives,
    environment markers, blank lines, and comment-only lines.
    """
    entries: list[DependencyEntry] = []
    idx = 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        line = re.split(r"\s*#", line)[0]
        line = re.split(r"\s*;", line)[0].strip()

        m = _REQ_LINE.match(line)
        if not m:
            continue

        name = m.group(1).strip()
        spec_str = m.group(2) or ""
        version = _parse_version_spec(spec_str)

        lookup_key = name.lower().replace("_", "-")
        spdx = PYPI_LICENSE_MAP.get(lookup_key) or PYPI_LICENSE_MAP.get(name)

        dep_id = f"pypi_{idx:03d}"
        entries.append(DependencyEntry(
            dep_id=dep_id,
            name=name,
            version=version,
            declared_license=spdx,
            resolved_license=spdx,
            is_direct=True,
            used_in="runtime",
        ))
        idx += 1

    return entries


# ---------------------------------------------------------------------------
# Core scan function
# ---------------------------------------------------------------------------

def scan_github_repo(github_url: str) -> dict[str, Any]:
    """
    Fetch package.json or requirements.txt from *github_url* and return
    structured findings JSON.

    Return shape::

        {
          "repo_url": str,
          "manifest_type": "npm" | "python" | None,
          "manifest_url": str | None,
          "dependencies": [...],
          "findings": [...],
          "policy": {...} | None,
          "grader_result": {...} | None,
          "summary": str,
        }
    """
    manifest_content: str | None = None
    manifest_url: str | None = None
    manifest_type: str | None = None

    BRANCHES = ["main", "master"]
    MANIFESTS = [
        ("package.json",     "npm"),
        ("requirements.txt", "python"),
    ]

    for branch in BRANCHES:
        for filename, mtype in MANIFESTS:
            try:
                raw_url = _github_url_to_raw(github_url, filename, branch)
            except ValueError as exc:
                return {
                    "repo_url": github_url,
                    "manifest_type": None,
                    "manifest_url": None,
                    "dependencies": [],
                    "findings": [],
                    "policy": None,
                    "grader_result": None,
                    "summary": str(exc),
                }
            content = _fetch_raw(raw_url)
            if content is not None:
                manifest_content = content
                manifest_url = raw_url
                manifest_type = mtype
                break
        if manifest_content is not None:
            break

    if manifest_content is None:
        return {
            "repo_url": github_url,
            "manifest_type": None,
            "manifest_url": None,
            "dependencies": [],
            "findings": [],
            "policy": None,
            "grader_result": None,
            "summary": (
                "No package.json or requirements.txt found in repository "
                "(tried main/master branches)."
            ),
        }

    project_license = "PROPRIETARY"
    try:
        if manifest_type == "npm":
            deps = parse_package_json(manifest_content, project_license)
        else:
            deps = parse_requirements_txt(manifest_content, project_license)
    except ValueError as exc:
        return {
            "repo_url": github_url,
            "manifest_type": manifest_type,
            "manifest_url": manifest_url,
            "dependencies": [],
            "findings": [],
            "policy": None,
            "grader_result": None,
            "summary": f"Parse error: {exc}",
        }

    policy = PolicyConfig(
        project_license=project_license,
        allowed_categories=[LicenseCategory.PERMISSIVE, LicenseCategory.PUBLIC_DOMAIN],
        forbidden_licenses=["GPL-2.0-only", "GPL-3.0-only", "AGPL-3.0-only"],
        copyleft_allowed_in_dev=True,
        network_copyleft_forbidden=True,
        distribution_type="proprietary",
    )

    findings: list[ScanFinding] = []
    ground_truth: dict[str, dict] = {}
    ecosystem = "npm" if manifest_type == "npm" else "python"
    ai_resolved_count = 0

    for dep in deps:
        spdx = dep.resolved_license or "UNKNOWN"
        ai_reasoning: str | None = None
        confidence = 0.85  # default for rule-based hits

        # ── Phase 2: Claude AI fallback ──────────────────────────────────
        # If the static map has no entry for this package, ask Claude.
        if spdx == "UNKNOWN":
            confidence = 0.0  # signal that we have no rule-based answer
            ai_result = _claude_classify_for_scan(dep.name, ecosystem)
            ai_reasoning = ai_result["reasoning"]
            if ai_result["spdx_id"] != "UNKNOWN":
                spdx = ai_result["spdx_id"]
                dep.resolved_license = spdx
                confidence = 0.70  # AI-inferred, lower confidence than static map
                ai_resolved_count += 1
        # ─────────────────────────────────────────────────────────────────

        severity_str = get_conflict_severity(project_license, spdx, "proprietary")
        category_str = _resolve_license_category(spdx)

        try:
            category = LicenseCategory(category_str)
        except ValueError:
            category = LicenseCategory.UNKNOWN

        try:
            severity = ConflictSeverity(severity_str)
        except ValueError:
            severity = ConflictSeverity.NONE

        reason: str | None = None
        remediation: str | None = None
        if severity != ConflictSeverity.NONE:
            reason = (
                f"{spdx} is incompatible with a {project_license} project: "
                "copyleft terms require derivative works to be open-sourced."
            )
            remediation = (
                f"Replace or isolate {dep.name} to avoid {spdx} copyleft obligations."
            )

        finding = ScanFinding(
            finding_id=str(_uuid.uuid4())[:8],
            source_id=dep.dep_id,
            license_spdx=spdx,
            category=category,
            severity=severity,
            conflict_reason=reason,
            remediation=remediation,
            agent_confidence=confidence,
            reviewed=True,
            ai_reasoning=ai_reasoning,
        )
        findings.append(finding)
        ground_truth[dep.dep_id] = {
            "resolved_license": spdx,
            "conflict_severity": severity_str,
            "is_conflict": severity != ConflictSeverity.NONE,
        }

    from graders.grader_medium import score as medium_score
    grader_result = medium_score(
        findings=findings,
        ground_truth=ground_truth,
        steps_used=len(deps),
        max_steps=max(30, len(deps) * 2),
    )

    conflict_count = sum(1 for f in findings if f.severity != ConflictSeverity.NONE)
    unknown_count  = sum(1 for f in findings if f.license_spdx == "UNKNOWN")
    pkg_label = "npm" if manifest_type == "npm" else "PyPI"
    ai_note = f", {ai_resolved_count} resolved by AI" if ai_resolved_count else ""
    summary = (
        f"Scanned {len(deps)} {pkg_label} "
        f"{'package' if len(deps) == 1 else 'packages'} from {github_url}. "
        f"Found {conflict_count} license conflict{'s' if conflict_count != 1 else ''}"
        + (f", {unknown_count} with unknown licenses" if unknown_count else "")
        + ai_note + "."
    )

    return {
        "repo_url": github_url,
        "manifest_type": manifest_type,
        "manifest_url": manifest_url,
        "dependencies": [d.model_dump() for d in deps],
        "findings": [f.model_dump() for f in findings],
        "policy": policy.model_dump(),
        "grader_result": grader_result.model_dump(),
        "summary": summary,
    }
