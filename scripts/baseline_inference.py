#!/usr/bin/env python3
"""
Baseline inference script using OpenAI API client.
Runs a model against all 3 tasks and reports reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/baseline_inference.py [--model gpt-4o] [--seed 42]
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from env.environment import LicenseComplianceEnv
from env.models import Action, ActionType, LicenseCategory, ConflictSeverity, ComplianceReport

TASKS = ["classify_licenses", "detect_conflicts", "generate_compliance_report"]

SYSTEM_PROMPT = """You are an expert software license compliance analyst.
You receive observations from a license compliance scanning environment and must respond
with a single JSON action object to complete the task.

Available action_type values:
- classify_license: fields: target_id, classification (SPDX), category, confidence, reason
- flag_conflict: fields: target_id, classification, category, severity, reason, remediation
- add_finding: fields: target_id, classification, category, severity, reason, remediation
- propose_remediation: fields: target_id, remediation
- generate_report: fields: report {summary, total_dependencies, compliant_count, violation_count, unknown_count, recommended_actions, overall_risk, agent_notes}
- mark_reviewed: fields: target_id

SPDX IDs: MIT, Apache-2.0, BSD-3-Clause, GPL-2.0-only, GPL-3.0-only, LGPL-2.1-only, AGPL-3.0-only, MPL-2.0, ISC, CC0-1.0, Unlicense, BUSL-1.1, PROPRIETARY
Categories: permissive, copyleft_weak, copyleft_strong, copyleft_network, proprietary, public_domain, unknown
Severities: none, low, medium, high, critical

Be systematic. Process each item one by one. For generate_compliance_report task, end with generate_report action.
Always output exactly one JSON object."""


def obs_to_prompt(obs_dict: dict) -> str:
    task, step, max_steps = obs_dict["task_id"], obs_dict["step"], obs_dict["max_steps"]
    lines = [f"TASK: {task} | Step {step}/{max_steps}", ""]
    msgs = obs_dict.get("messages", [])
    if msgs:
        lines += ["Messages: " + " | ".join(msgs[-2:]), ""]
    policy = obs_dict.get("policy")
    if policy:
        lines += [f"POLICY: project={policy['project_license']}, dist={policy['distribution_type']}, "
                  f"forbidden={policy.get('forbidden_licenses', [])}", ""]
    findings = obs_dict.get("scan_results", [])
    reviewed = {f["source_id"] for f in findings}
    files = [f for f in obs_dict.get("files_to_scan", []) if not f.get("reviewed") and f["file_id"] not in reviewed]
    if files:
        lines.append(f"UNCLASSIFIED FILES ({len(files)}):")
        for f in files[:2]:
            lines += [f"  [{f['file_id']}] {f['path']}", f"  {f['content_snippet'][:250]}...", ""]
    deps = [d for d in obs_dict.get("dependencies", []) if d["dep_id"] not in reviewed]
    if deps:
        lines.append(f"UNREVIEWED DEPS ({len(deps)}):")
        for d in deps[:4]:
            lines.append(f"  [{d['dep_id']}] {d['name']}=={d['version']} license={d.get('declared_license','?')} used_in={d.get('used_in','runtime')}")
        lines.append("")
    if not files and not deps:
        crit = len([f for f in findings if f.get("severity") == "critical"])
        high = len([f for f in findings if f.get("severity") == "high"])
        lines.append(f"All items reviewed. Findings: {len(findings)} total, {crit} critical, {high} high.")
        if task == "generate_compliance_report":
            lines.append("Now call generate_report action with a structured ComplianceReport.")
    return "\n".join(lines)


def call_llm(client, model, history, prompt):
    history.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        response_format={"type": "json_object"},
        temperature=0.1, max_tokens=800,
    )
    content = resp.choices[0].message.content
    history.append({"role": "assistant", "content": content})
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"action_type": "mark_reviewed", "target_id": "unknown"}


def parse_action(d: dict) -> Action:
    try:
        at = ActionType(d.get("action_type", "mark_reviewed"))
    except ValueError:
        at = ActionType.MARK_REVIEWED
    report = None
    if d.get("report"):
        r = d["report"]
        try:
            report = ComplianceReport(
                summary=r.get("summary", ""),
                total_dependencies=int(r.get("total_dependencies", 0)),
                compliant_count=int(r.get("compliant_count", 0)),
                violation_count=int(r.get("violation_count", 0)),
                unknown_count=int(r.get("unknown_count", 0)),
                recommended_actions=r.get("recommended_actions", []),
                overall_risk=ConflictSeverity(r.get("overall_risk", "medium")),
                agent_notes=r.get("agent_notes", ""),
            )
        except Exception:
            pass
    cat, sev = None, None
    try:
        if d.get("category"): cat = LicenseCategory(d["category"])
    except ValueError: pass
    try:
        if d.get("severity"): sev = ConflictSeverity(d["severity"])
    except ValueError: pass
    return Action(
        action_type=at, target_id=d.get("target_id"),
        classification=d.get("classification"), category=cat, severity=sev,
        confidence=float(d.get("confidence", 0.8)),
        reason=d.get("reason"), remediation=d.get("remediation"), report=report,
    )


def run_task(client, model, task_id, seed):
    print(f"\n{'='*60}\nTASK: {task_id}  |  seed={seed}  |  model={model}\n{'='*60}")
    env = LicenseComplianceEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    history, done, step = [], False, 0
    while not done:
        prompt = obs_to_prompt(obs.model_dump())
        try:
            action_dict = call_llm(client, model, history, prompt)
        except Exception as e:
            print(f"  [LLM ERROR] {e}"); break
        try:
            action = parse_action(action_dict)
        except Exception as e:
            print(f"  [PARSE ERROR] {e}"); continue
        obs, reward, done, info = env.step(action)
        step += 1
        print(f"  Step {step:2d} | {action.action_type.value:25s} | target={action.target_id or '-':15s} | reward={reward.total:+.3f} | {reward.message[:55]}")
        time.sleep(0.25)
    final = env.final_score()
    print(f"\n  FINAL: {final.total:.4f} | {final.message}")
    print(f"  Breakdown: {json.dumps(final.breakdown, indent=2)}")
    return {"task_id": task_id, "seed": seed, "model": model, "steps": step, "final_score": final.total, "breakdown": final.breakdown}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="all", choices=["all"] + TASKS)
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY"); sys.exit(1)
    client = OpenAI(api_key=api_key)
    tasks = TASKS if args.task == "all" else [args.task]
    results = [run_task(client, args.model, t, args.seed) for t in tasks]
    print(f"\n{'='*60}\nBASELINE RESULTS\n{'='*60}")
    print(f"{'Task':<38} {'Score':>8} {'Steps':>6}")
    print('-'*55)
    for r in results:
        print(f"{r['task_id']:<38} {r['final_score']:>8.4f} {r['steps']:>6}")
    avg = sum(r["final_score"] for r in results) / len(results)
    print(f"{'AVERAGE':<38} {avg:>8.4f}")
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to baseline_results.json")

if __name__ == "__main__":
    main()