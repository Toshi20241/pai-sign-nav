"""Matched-condition comparison of image-based and simple baseline policies."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from mission import run_episode
from vision import SignClassifier


def run_benchmark(out_dir=None, count=6, seed_start=81000):
    out = Path(out_dir or Path(__file__).parent/"results"/"benchmark")
    out.mkdir(parents=True, exist_ok=True)
    model = SignClassifier()
    rows = []
    start_time = time.perf_counter()
    for seed in range(seed_start, seed_start+count):
        for policy in ("learned", "straight"):
            r = run_episode(seed, policy, "standard", model=model, record=False)["result"]
            rows.append(r)
    with (out/"trials.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    flat_fields = ["seed", "policy", "profile", "completed", "success", "contacts",
                   "duration_seconds", "path_length_m", "correct_decisions",
                   "decision_count", "all_signs_followed", "max_wheel_acceleration"]
    with (out/"trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"seed_start": seed_start, "conditions": count,
               "total_runs": len(rows), "elapsed_seconds": time.perf_counter()-start_time,
               "profile": "standard", "policies": {}}
    for policy in ("learned", "straight"):
        trials = [r for r in rows if r["policy"] == policy]
        summary["policies"][policy] = {
            "runs": len(trials), "completed": sum(r["completed"] for r in trials),
            "successes": sum(r["success"] for r in trials),
            "all_signs_followed": sum(r["all_signs_followed"] for r in trials),
            "correct_decisions": sum(r["correct_decisions"] for r in trials),
            "decision_count": sum(r["decision_count"] for r in trials),
            "contact_events": sum(r["contacts"] for r in trials),
            "mean_duration_seconds": float(np.mean([r["duration_seconds"] for r in trials])),
            "mean_path_length_m": float(np.mean([r["path_length_m"] for r in trials]))}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=81000)
    args = parser.parse_args()
    run_benchmark(args.out, args.count, args.seed_start)
