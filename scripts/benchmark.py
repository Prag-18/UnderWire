#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import statistics

from openai import OpenAI

from baseline_inference import TASKS, run_task


def run_benchmark(task: str = "detect_conflicts", seeds: int = 50, model: str = "gpt-4o") -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running the benchmark.")

    client = OpenAI(api_key=api_key)
    tasks = TASKS if task == "all" else [task]
    summaries = []

    for task_id in tasks:
        scores = []
        for seed in range(seeds):
            print(f"Running task={task_id} seed={seed} model={model}...")
            result = run_task(client, model, task_id, seed)
            scores.append(result["final_score"])

        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        summary = {
            "task": task_id,
            "seeds": seeds,
            "model": model,
            "mean_score": mean_score,
            "std_dev": std_dev,
        }
        summaries.append(summary)

        print("\n========== BENCHMARK RESULTS ==========")
        print(f"Task: {task_id}")
        print(f"Seeds: {seeds}")
        print(f"Mean Score: {mean_score:.3f}")
        print(f"Std Dev: {std_dev:.3f}")
        print("=======================================")

    return {"results": summaries}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="detect_conflicts", choices=["all"] + TASKS)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    try:
        run_benchmark(task=args.task, seeds=args.seeds, model=args.model)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
