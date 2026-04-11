---
title: License Compliance Scanner (OpenEnv)
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# License Compliance Scanner (OpenEnv)

## Overview

Software license compliance is a critical engineering domain where open-source adoption relies heavily on ensuring copyleft restrictions don't unintentionally breach proprietary distribution systems. The License Compliance Scanner provides an agentic evaluation environment designed to test and train AI autonomous agents inside simulated codebases.

## Features

- **Deterministic Scenarios**: Identical initial states upon loading unique seeds.
- **Brier Scoring**: Evaluates models on their confidence calibration mapping to classification outcomes.
- **Deceptive License Files**: Analyzes files claiming standard boilerplate components masking more aggressive terms.
- **SQLite Persistence**: Complete state persistence mapping observation histories.
- **Multi-task Evaluation**: Modular assessment structures simulating distinct compliance duties.

## Tasks

### Easy (classify_licenses)
- **Objective:** Classify individual files based on their active headers and body implementations.
- **Challenge:** Deceptive files present mismatches resolving testing resilience.

### Medium (detect_conflicts)
- **Objective:** Evaluate dependency manifests determining topological collisions.
- **Challenge:** Leverages F1 evaluation metrics analyzing precision & recall.

### Hard (generate_compliance_report)
- **Objective:** Build full analysis encompassing vendor, deep dependencies, and mixed files under constraints.
- **Challenge:** Heavily constrained SaaS restrictions + Unknown License classification.

## Action Space

- `classify_license`: Submit SPDX class per Target.
- `flag_conflict`: Mark severity definitions for project architectures.
- `mark_reviewed`: Clean standard safe elements.
- `request_clarification`: Safe fallback against ambiguous models.
- `add_finding`: Inject human/contextual findings.
- `propose_remediation`: Implement strategy fixes for high-bound issues.
- `generate_report`: Render formal structured compilation.

## Observation Space

- **Dependencies**: Explicit definition versions, stated terms.
- **Files**: Relative paths mapping back to contextual snippets bounding exact text fragments.
- **Policy**: Baseline compliance config dictating project stance definitions.
- **Findings**: Tracked step outputs recorded by the agent iteratively interacting in the environment.

## Reward System

- **Step Feedback**: Direct partial credits validating heuristic successes executing correctly formed observations.
- **Final Scoring**: Score range is **0.0 – 1.0** for Easy and Medium tasks. The Hard task (`generate_compliance_report`) is intentionally **capped at 0.80** to model irreducible uncertainty in real-world vendor audits.

## Setup Instructions

### Local
```bash
cp .env.example .env        # add your OPENAI_API_KEY inside .env
pip install -r requirements.txt
uvicorn server:app --reload
```

### Docker
```bash
docker build -t license-env .
docker run -p 7860:7860 license-env
```

## API Usage

Example flows map interacting directly with the agent environments tracking REST schemas:

- `POST /env/create`: Instantiates the configuration via requested seeds and bounds.
- `POST /env/step`: Enacts a single structured `Action` via JSON generating reward increments.
- `GET /env/score/{session_id}`: Compiles and checks final run status limits.

## Baseline Results

| Task | Score |
| :--- | :--- |
| classify_licenses | 0.72 |
| detect_conflicts | 0.58 |
| generate_compliance_report | 0.41 |

## Hugging Face Deployment

Configuration supports deployment under standard Docker space environments tracking specific Hugging Face configurations running via container mapping standard openenv tag parameters.

## Future Improvements

- Multi-repo scanning simulations.
- UI dashboard visualizer integrating step observations dynamically in frontends.
- Broadening training constraints mapping stronger multi-agent routines.
