"""Run the complete vision-to-wheel-control loop and save auditable logs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from controller import (BranchPolicy, DifferentialDrive, alignment_command,
                        waypoint_command)
from world import GOAL, JUNCTION_Y, PracticeWorld


def run_episode(seed=20261029, policy="learned", profile="standard", save_dir=None,
                model=None, record=True):
    world = PracticeWorld(seed, profile)
    robot = DifferentialDrive(world.start, world.wheel_gain)
    selector = BranchPolicy(policy, model=model)
    out = Path(save_dir) if save_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out/"camera").mkdir(exist_ok=True)
    dt, control_dt = .01, .05
    control_every = round(control_dt/dt)
    t, step_index = 0., 0
    stage, junction, path_index = "APPROACH", 0, 0
    path = [[0., JUNCTION_Y[0]]]
    linear, angular, last_v, last_omega = 0., 0., 0., 0.
    read_start = None
    choices, observations, events, telemetry = [], [], [], []
    contacts = 0
    reached_goal = False
    active_observation = -1

    def event(kind, **kwargs):
        events.append(dict(time=round(t, 3), event=kind, junction=junction, **kwargs))

    event("START")
    while t < 150.:
        if step_index % control_every == 0:
            if stage in ("APPROACH", "FOLLOW_BRANCH", "GOAL"):
                linear, angular, arrived = waypoint_command(robot.x, robot.y, robot.theta, path[path_index])
                if arrived:
                    path_index += 1
                    if path_index < len(path):
                        linear, angular, _ = waypoint_command(robot.x, robot.y, robot.theta, path[path_index])
                    elif stage == "APPROACH":
                        stage = "ALIGN"
                        linear = angular = 0.
                        event("ARRIVE_SIGN")
                    elif stage == "FOLLOW_BRANCH":
                        junction += 1
                        if junction < len(JUNCTION_Y):
                            path = [[0., JUNCTION_Y[junction]]]
                            stage = "APPROACH"
                        else:
                            path = [list(GOAL)]
                            stage = "GOAL"
                        path_index = 0
                    else:
                        stage = "FINISH"
                        linear = angular = 0.
                        event("GOAL_REACHED")
            elif stage == "ALIGN":
                linear, angular, aligned = alignment_command(robot.theta)
                if aligned and abs(last_v) < .008 and abs(last_omega) < .08:
                    stage = "READ_SIGN"
                    linear = angular = 0.
                    read_start = t
                    image = world.capture_sign(junction)
                    prediction = selector.choose(image)
                    chosen = int(prediction["label"])
                    choices.append(chosen)
                    active_observation = junction
                    image_name = f"camera/sign_{junction+1:02d}.png"
                    if out is not None:
                        image.save(out/image_name)
                    observations.append({"index": junction, "time": round(t, 3),
                                         "image": image_name, "prediction": prediction,
                                         "robot_pose": [robot.x, robot.y, robot.theta]})
                    event("CLASSIFY", choice=chosen, prediction=prediction)
            elif stage == "READ_SIGN":
                linear = angular = 0.
                if t-read_start >= 1.6:
                    path = world.branch_points(JUNCTION_Y[junction], choices[-1])
                    path_index = 0
                    stage = "FOLLOW_BRANCH"
                    event("FOLLOW_BRANCH", choice=choices[-1])
            elif stage == "FINISH":
                linear = angular = 0.
                if abs(last_v) < .001 and abs(last_omega) < .01:
                    reached_goal = math.hypot(robot.x-GOAL[0], robot.y-GOAL[1]) < .08
                    event("COMPLETE")
                    break
            if record or out is not None:
                telemetry.append({"time": round(t, 4), "x": robot.x, "y": robot.y,
                                  "theta": robot.theta, "wheel_left": robot.left,
                                  "wheel_right": robot.right, "linear": last_v,
                                  "angular": last_omega, "command_linear": linear,
                                  "command_angular": angular, "state": stage,
                                  "junction": min(junction, 2),
                                  "observation_index": active_observation,
                                  "choice": choices[-1] if choices else -1})
        last_v, last_omega = robot.step(linear, angular, dt)
        if world.collides(robot.x, robot.y):
            contacts += 1
            event("CONTACT")
            break
        t += dt
        step_index += 1
    if not reached_goal and not contacts:
        event("TIME_LIMIT")
    evaluation = world.evaluate(choices)
    result = {"seed": int(seed), "policy": policy, "profile": profile,
              "completed": bool(reached_goal),
              "success": bool(reached_goal and evaluation["all_signs_followed"] and contacts == 0),
              "contacts": contacts, "duration_seconds": round(t, 3),
              "path_length_m": robot.path_length, "choices": choices,
              "final_position": [robot.x, robot.y],
              "max_wheel_acceleration": robot.max_observed_acceleration,
              **evaluation}
    mission = {"schema_version": 1, "result": result, "simulation": {
                   "model": "no-slip differential-drive planar kinematics",
                   "integration_step_seconds": dt, "control_step_seconds": control_dt,
                   "sensor": "synthetic 96px sign panel, known capture stations",
                   "localization": "exact simulator pose",
                   "collision": "circle-versus-rectangle geometry; terminate on contact",
                   "wheel_radius_m": robot.radius, "axle_m": robot.axle,
                   "wheel_tau_seconds": robot.tau,
                   "wheel_acceleration_limit_rad_s2": robot.acceleration_limit,
                   "wheel_gain": list(world.wheel_gain)},
               "observations": observations, "events": events}
    if out is not None:
        (out/"world.json").write_text(json.dumps(world.public_geometry(), indent=2), encoding="utf-8")
        (out/"mission.json").write_text(json.dumps(mission, indent=2), encoding="utf-8")
        if telemetry:
            with (out/"telemetry.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(telemetry[0]))
                writer.writeheader()
                writer.writerows(telemetry)
    return {"result": result, "mission": mission, "telemetry": telemetry,
            "world": world.public_geometry()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--policy", choices=["learned", "straight"], default="learned")
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--out", default=str(Path(__file__).parent/"results"/"demo"))
    args = parser.parse_args()
    result = run_episode(args.seed, args.policy, args.profile, args.out)
    print(json.dumps(result["result"], indent=2))


if __name__ == "__main__":
    main()

