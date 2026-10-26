"""Verify the saved motion, camera decisions, and comparison bookkeeping."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vision import SignClassifier
from world import PracticeWorld


def verify():
    root = Path(__file__).parent
    demo = root/"results"/"demo"
    mission = json.loads((demo/"mission.json").read_text())
    with (demo/"telemetry.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    time = np.array([float(r["time"]) for r in rows])
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    xy_steps = np.hypot(np.diff(x), np.diff(y))
    wheels = np.array([[float(r["wheel_left"]), float(r["wheel_right"])] for r in rows])
    wheel_accel = np.max(np.abs(np.diff(wheels, axis=0)/np.diff(time)[:, None]))
    robot_acc_limit = mission["simulation"]["wheel_acceleration_limit_rad_s2"]
    world = PracticeWorld(mission["result"]["seed"])
    classifier = SignClassifier()
    image_labels = [classifier.predict(Image.open(demo/o["image"]))["label"]
                    for o in mission["observations"]]
    # Counterfactual image input: cyclically rearranged observed images must
    # rearrange decisions without changing geometry or class-to-route mapping.
    reordered = mission["observations"][1:] + mission["observations"][:1]
    swapped_labels = [classifier.predict(Image.open(demo/o["image"]))["label"]
                      for o in reordered]
    benchmark = json.loads((root/"results"/"benchmark"/"trials.json").read_text())
    summary = json.loads((root/"results"/"benchmark"/"summary.json").read_text())
    checks = {
        "positive_uniform_telemetry_time": bool(np.allclose(np.diff(time), .05)),
        "position_is_continuous": bool(np.all(xy_steps <= .32*np.diff(time)+1e-7)),
        "wheel_acceleration_within_bound": bool(wheel_accel <= robot_acc_limit+1e-6),
        "telemetry_has_no_obstacle_overlap": bool(not any(world.collides(a,b) for a,b in zip(x,y))),
        "saved_images_reproduce_decisions": image_labels == mission["result"]["choices"],
        "changed_images_change_actions": swapped_labels == image_labels[1:]+image_labels[:1]
                                         and swapped_labels != image_labels,
        "left_and_right_detours_visible_in_positions": bool(x.min() < -1.2 and x.max() > 1.2),
        "all_benchmark_conditions_retained": len(benchmark) == summary["conditions"]*2,
        "policy_conditions_are_matched": all(
             len({r["seed"] for r in benchmark if r["policy"] == p}) == summary["conditions"]
             for p in ("learned", "straight")),
        "successful_trials_follow_all_signs": all(
            not r["success"] or (r["completed"] and r["all_signs_followed"] and r["contacts"] == 0)
            for r in benchmark)
    }
    report = {"checks": checks, "passed": sum(checks.values()), "total": len(checks),
              "max_telemetry_step_m": float(xy_steps.max()),
              "max_observed_wheel_acceleration_rad_s2": float(wheel_accel),
              "original_image_decisions": image_labels,
              "reordered_image_decisions": swapped_labels}
    (root/"results"/"navigation_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit("Navigation verification failed")
    return report


if __name__ == "__main__":
    verify()
