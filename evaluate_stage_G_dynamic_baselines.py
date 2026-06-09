"""Evaluate Stage G dynamic scenarios with multiple baselines.

This script does not train models. It reconstructs the dynamic evaluation
scenarios described by Stage G HGCR-PPO manifests and evaluates rule baselines
on the same scenario ids. HGCR-PPO is replayed from the run checkpoint when
available.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

from schedule_validator import validate_schedule
from src.baselines.heuristics import (
    candidate_load,
    choose_split_num,
    estimated_completion_time,
    lookahead_score,
    mean_candidate_processing_time,
)
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.evaluation.metrics import compute_metrics


RUNS_DIR = Path("data/results/stage_G/hgcr_dynamic_ppo/runs")
OUTPUT_DIR = Path("data/results/stage_G/baseline_eval")
DEFAULT_METHODS = ["Random", "SPT", "LPT", "FIFO", "GreedyECT", "Lookahead", "MinLoad", "MLP-Ranker", "HGCR-PPO"]
DETAIL_FIELDS = [
    "scenario_run_id",
    "arrival_intensity",
    "carryover_ratio",
    "reward_beta",
    "seed",
    "instance_id",
    "method",
    "Cmax",
    "average_completion_time",
    "average_waiting_time",
    "machine_utilization",
    "load_balance_std",
    "runtime_seconds",
    "valid_schedule",
]
SUMMARY_FIELDS = [
    "arrival_intensity",
    "carryover_ratio",
    "reward_beta",
    "seed",
    "method",
    "Cmax_mean",
    "Cmax_std",
    "Cmax_min",
    "Cmax_max",
    "average_completion_time_mean",
    "average_waiting_time_mean",
    "machine_utilization_mean",
    "load_balance_std_mean",
    "runtime_mean",
    "valid_schedule_rate",
]
TRACE_FIELDS = [
    "scenario_run_id",
    "arrival_intensity",
    "carryover_ratio",
    "reward_beta",
    "seed",
    "instance_id",
    "method",
    "job_id",
    "batch_id",
    "process_type",
    "machine_id",
    "start_time",
    "end_time",
    "duration",
    "split_ratio",
    "rolling_period",
]


def run_id() -> str:
    return f"stageG_baselines_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def first_file(run_dir: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_manifest(run_dir: Path) -> dict | None:
    path = first_file(run_dir, ["manifest.json", "manifest__*.json", "manifest*.json"])
    if path is None:
        print(f"Warning: skip {run_dir}, missing manifest.")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Warning: skip {run_dir}, malformed manifest.")
        return None


def valid_manifest(manifest: dict) -> bool:
    return (
        manifest.get("stage") == "G"
        and manifest.get("experiment_family") == "hgcr_dynamic_ppo"
        and manifest.get("scenario_type") == "dynamic_rolling"
    )


def select_stage_g_runs(runs_dir: Path, max_runs: int | None) -> List[tuple[Path, dict]]:
    if not runs_dir.exists():
        print(f"Warning: runs_dir does not exist: {runs_dir}")
        return []
    selected = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        manifest = read_manifest(run_dir)
        if manifest is None:
            continue
        if not valid_manifest(manifest):
            print(f"Warning: skip {run_dir}, not a clean Stage G dynamic run.")
            continue
        selected.append((run_dir, manifest))
        if max_runs is not None and len(selected) >= max_runs:
            break
    return selected


def reset_env_for_scenario(scenario: dict) -> RollingSchedulingEnv:
    env = RollingSchedulingEnv(scenario["instance"])
    env.reset(scenario["instance"])
    env.machine_available_time.update({k: float(v) for k, v in scenario["machine_initial_available_time"].items()})
    env.current_cmax = max(env.machine_available_time.values(), default=0.0)
    return env


def choose_job(env: RollingSchedulingEnv, method: str, rng: random.Random, ranker_model=None, device="cpu", top_k: int = 5) -> str:
    jobs = env.get_schedulable_jobs()
    if method == "Random":
        return rng.choice(jobs)
    if method == "SPT":
        return min(jobs, key=lambda j: (mean_candidate_processing_time(env, j), j))
    if method == "LPT":
        return max(jobs, key=lambda j: (mean_candidate_processing_time(env, j), j))
    if method == "FIFO":
        return min(jobs, key=lambda j: (env.job_by_id[j].release_time, j))
    if method == "GreedyECT":
        return min(jobs, key=lambda j: (estimated_completion_time(env, j), j))
    if method == "Lookahead":
        return min(jobs, key=lambda j: (lookahead_score(env, j), j))
    if method in {"MinLoad", "MinCandidateLoad"}:
        return min(jobs, key=lambda j: (candidate_load(env, j), env.job_by_id[j].release_time, j))
    if method == "MLP-Ranker" and ranker_model is not None:
        import torch
        from candidate_generator import generate_candidates
        from stage_c_utils import extract_candidate_features

        candidates = generate_candidates(env, candidate_mode="hybrid_topk", top_k=top_k, fallback_to_all=True)
        features = torch.tensor([extract_candidate_features(env, candidates)], dtype=torch.float32, device=device)
        with torch.no_grad():
            scores = ranker_model(features).squeeze(0).detach().cpu().numpy()
        return candidates[int(scores.argmin())]
    if method == "MLP-Ranker":
        return min(jobs, key=lambda j: (estimated_completion_time(env, j), j))
    raise ValueError(f"Unsupported method: {method}")


def rollout_rule(scenario: dict, method: str, seed: int, ranker_model=None, device="cpu", top_k: int = 5):
    rng = random.Random(seed)
    env = reset_env_for_scenario(scenario)
    start = time.perf_counter()
    while not env.is_done():
        job_id = choose_job(env, method, rng, ranker_model=ranker_model, device=device, top_k=top_k)
        env.step((job_id, choose_split_num(env, job_id)))
    return env, time.perf_counter() - start


def load_ranker(path: str, device: str):
    if not Path(path).exists():
        print(f"Warning: ranker checkpoint not found, MLP-Ranker falls back to GreedyECT: {path}")
        return None
    from mlp_models import load_checkpoint

    return load_checkpoint(Path(path), device=device)


def load_hgcr_policy(run_dir: Path, manifest: dict, device: str):
    import torch
    from run_hgcr_dynamic_ppo import RULE_NAMES, RuleActorCritic, load_ranker_or_none

    ckpt_path = first_file(run_dir, ["hgcr_dynamic_ppo.pt", "hgcr_dynamic_ppo__*.pt"])
    if ckpt_path is None:
        print(f"Warning: missing HGCR checkpoint in {run_dir}; HGCR-PPO replay skipped.")
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RuleActorCritic(int(ckpt["state_dim"]), len(ckpt.get("rule_names", RULE_NAMES))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    args = SimpleNamespace(**(manifest.get("args") or {}))
    ranker = load_ranker_or_none(getattr(args, "ranker_ckpt", ""), device)
    return model, ranker


def rollout_hgcr(scenario: dict, model, ranker_model, manifest: dict, device: str):
    import torch
    from run_hgcr_dynamic_ppo import RULE_NAMES, rule_choices, state_features

    args = SimpleNamespace(**(manifest.get("args") or {}))
    top_k = int(getattr(args, "top_k", manifest.get("top_k", 5)))
    env = reset_env_for_scenario(scenario)
    start = time.perf_counter()
    while not env.is_done():
        choices = rule_choices(env, ranker_model=ranker_model, device=device, top_k=top_k)
        state = state_features(env, choices, ranker_model is not None)
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist, _ = model.dist_value(state_t)
            action = int(torch.argmax(dist.probs, dim=-1).item())
        env.step((choices[action], choose_split_num(env, choices[action])))
    return env, time.perf_counter() - start


def detail_row(run_manifest: dict, scenario: dict, method: str, env, runtime: float) -> dict:
    metrics = compute_metrics(env)
    valid = validate_schedule(env, scenario["instance"])["is_valid_schedule"]
    return {
        "scenario_run_id": run_manifest["run_id"],
        "arrival_intensity": run_manifest["arrival_intensity"],
        "carryover_ratio": run_manifest["carryover_ratio"],
        "reward_beta": run_manifest["reward_beta"],
        "seed": run_manifest["seed"],
        "instance_id": scenario["scenario_id"],
        "method": "MinLoad" if method == "MinCandidateLoad" else method,
        "Cmax": metrics["Cmax_roll"],
        "average_completion_time": metrics["average_completion_time"],
        "average_waiting_time": metrics["average_waiting_time"],
        "machine_utilization": metrics["machine_utilization"],
        "load_balance_std": metrics["load_balance_std"],
        "runtime_seconds": runtime,
        "valid_schedule": valid,
    }


def schedule_trace_rows(run_manifest: dict, scenario: dict, method: str, env) -> List[dict]:
    period = float(getattr(scenario["instance"], "rolling_period_length", 1.0) or 1.0)
    rows = []
    for subtask in env.subtasks:
        rows.append(
            {
                "scenario_run_id": run_manifest["run_id"],
                "arrival_intensity": run_manifest["arrival_intensity"],
                "carryover_ratio": run_manifest["carryover_ratio"],
                "reward_beta": run_manifest["reward_beta"],
                "seed": run_manifest["seed"],
                "instance_id": scenario["scenario_id"],
                "method": method,
                "job_id": subtask.job_id,
                "batch_id": scenario["scenario_id"],
                "process_type": subtask.process_type,
                "machine_id": subtask.machine_id,
                "start_time": subtask.start_time,
                "end_time": subtask.completion_time,
                "duration": subtask.duration,
                "split_ratio": subtask.ratio,
                "rolling_period": int(subtask.start_time // period),
            }
        )
    return rows


def summarize(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in rows:
        key = (row["arrival_intensity"], row["carryover_ratio"], row["reward_beta"], row["seed"], row["method"])
        groups.setdefault(key, []).append(row)
    out = []
    for (arrival, carryover, beta, seed, method), vals in sorted(groups.items()):
        cmax = [float(row["Cmax"]) for row in vals]
        out.append(
            {
                "arrival_intensity": arrival,
                "carryover_ratio": carryover,
                "reward_beta": beta,
                "seed": seed,
                "method": method,
                "Cmax_mean": mean(cmax),
                "Cmax_std": pstdev(cmax) if len(cmax) > 1 else 0.0,
                "Cmax_min": min(cmax),
                "Cmax_max": max(cmax),
                "average_completion_time_mean": mean(float(row["average_completion_time"]) for row in vals),
                "average_waiting_time_mean": mean(float(row["average_waiting_time"]) for row in vals),
                "machine_utilization_mean": mean(float(row["machine_utilization"]) for row in vals),
                "load_balance_std_mean": mean(float(row["load_balance_std"]) for row in vals),
                "runtime_mean": mean(float(row["runtime_seconds"]) for row in vals),
                "valid_schedule_rate": mean(1.0 if str(row["valid_schedule"]).lower() == "true" else 0.0 for row in vals),
            }
        )
    return out


def write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> Path | None:
    selected = select_stage_g_runs(Path(args.runs_dir), args.max_runs)
    eval_id = run_id()
    out_dir = Path(args.output_dir) / "runs" / eval_id
    print(f"Selected Stage G runs: {len(selected)}")
    for run_dir, manifest in selected:
        print(f"  - {manifest.get('run_id')} from {run_dir}")
    print(f"Planned output dir: {out_dir}")
    if args.dry_run:
        print("Dry run enabled: no baseline evaluation will be executed and no files will be written.")
        return out_dir

    methods = list(args.methods)
    device = args.device
    ranker_model = None if "MLP-Ranker" not in methods else load_ranker(args.ranker_ckpt, device)
    detail_rows: List[dict] = []
    trace_rows: List[dict] = []
    for run_dir, manifest in selected:
        from dynamic_rolling_scenarios import generate_dynamic_scenarios

        scenario_count = min(int(manifest.get("args", {}).get("eval_scenarios", 50)), args.max_instances or 10**9)
        scenarios = generate_dynamic_scenarios(
            manifest["size"],
            "test",
            scenario_count,
            int(manifest["seed"]) + 999,
            manifest["arrival_intensity"],
            manifest["carryover_ratio"],
            processing_time_noise=float(manifest.get("args", {}).get("processing_time_noise", 0.0)),
            machine_initial_load=str(manifest.get("args", {}).get("machine_initial_load", "low")),
        )
        hgcr_model = hgcr_ranker = None
        if "HGCR-PPO" in methods:
            hgcr_model, hgcr_ranker = load_hgcr_policy(run_dir, manifest, device)
        for scenario in scenarios:
            for method in methods:
                if method == "HGCR-PPO":
                    if hgcr_model is None:
                        continue
                    env, runtime = rollout_hgcr(scenario, hgcr_model, hgcr_ranker, manifest, device)
                else:
                    canonical = "MinCandidateLoad" if method == "MinLoad" else method
                    env, runtime = rollout_rule(scenario, canonical, int(manifest["seed"]), ranker_model=ranker_model, device=device, top_k=int(manifest["top_k"]))
                detail_rows.append(detail_row(manifest, scenario, method, env, runtime))
                if args.save_schedule_trace:
                    trace_rows.extend(schedule_trace_rows(manifest, scenario, method, env))

    summary_rows = summarize(detail_rows)
    if args.no_write:
        print(f"No-write enabled: evaluated {len(detail_rows)} detail rows and {len(trace_rows)} trace rows, no files written.")
        return out_dir
    write_csv(out_dir / f"baseline_eval_detail__{eval_id}.csv", detail_rows, DETAIL_FIELDS)
    write_csv(out_dir / f"baseline_eval_summary__{eval_id}.csv", summary_rows, SUMMARY_FIELDS)
    if args.save_schedule_trace:
        write_csv(out_dir / f"schedule_trace__{eval_id}.csv", trace_rows, TRACE_FIELDS)
    (out_dir / f"manifest__{eval_id}.json").write_text(
        json.dumps({"run_id": eval_id, "stage": "G", "args": vars(args), "source_runs": [m["run_id"] for _, m in selected]}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved baseline evaluation to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default=str(RUNS_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--ranker_ckpt", default="checkpoints/stage_C/mlp_ranker/small_topk5_soft_ce/best.pt")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--save_schedule_trace", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.smoke_test:
        args.max_runs = 1 if args.max_runs is None else min(args.max_runs, 1)
        args.max_instances = 1 if args.max_instances is None else min(args.max_instances, 1)
    run(args)


if __name__ == "__main__":
    main()
