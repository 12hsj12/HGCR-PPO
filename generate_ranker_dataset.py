"""Generate HybridTopK oracle ranking data for Stage C."""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
from src.baselines.heuristics import choose_split_num
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from stage_c_utils import (
    best_candidate_index,
    current_time,
    dataset_path,
    extract_candidate_features,
    fifo_first,
    hybrid_candidates,
    oracle_cmax_per_candidate,
    save_ranker_records,
)


def generate_records_for_instance(instance, top_k: int, oracle_rollout_policy: str = "fifo") -> List[Dict]:
    env = RollingSchedulingEnv(instance)
    env.reset(instance)
    records: List[Dict] = []
    step_id = 0

    while not env.is_done():
        candidates = hybrid_candidates(env, top_k)
        if not candidates:
            raise RuntimeError("HybridTopK returned an empty candidate set.")

        features = extract_candidate_features(env, candidates)
        cmax_values = oracle_cmax_per_candidate(env, candidates, rollout_policy=oracle_rollout_policy)
        best_idx = best_candidate_index(cmax_values)
        best_job = candidates[best_idx]
        baseline_job = fifo_first(env)

        records.append(
            {
                "instance_id": getattr(instance, "instance_id", instance.name),
                "step_id": step_id,
                "current_time": current_time(env),
                "current_period": int(current_time(env) // instance.rolling_period_length),
                "candidate_job_ids": list(candidates),
                "candidate_features": features,
                "candidate_scores": [-float(value) for value in cmax_values],
                "best_candidate_index": best_idx,
                "best_job_id": best_job,
                "oracle_cmax_per_candidate": [float(value) for value in cmax_values],
                "baseline_selected_job": baseline_job,
                "method_used": "hybrid_topk_oracle_debug",
                "oracle_rollout_policy": oracle_rollout_policy,
            }
        )
        env.step((best_job, choose_split_num(env, best_job)))
        step_id += 1

    return records


def generate_dataset(
    size: str,
    split: str,
    top_k: int = 5,
    max_instances: int | None = None,
    oracle_rollout_policy: str = "fifo",
    output_dir: str = "data/ranker_dataset/",
) -> List[Dict]:
    ensure_fixed_dataset([size], [split])
    instances = load_fixed_instances(size, split)
    if max_instances is not None:
        instances = instances[: max(0, max_instances)]

    records: List[Dict] = []
    started = time.perf_counter()
    for idx, instance in enumerate(instances, start=1):
        instance_records = generate_records_for_instance(instance, top_k, oracle_rollout_policy)
        records.extend(instance_records)
        print(
            f"[{idx}/{len(instances)}] {getattr(instance, 'instance_id', instance.name)} "
            f"states={len(instance_records)} total_states={len(records)}"
        )

    path = dataset_path(output_dir, size, split, top_k)
    save_ranker_records(records, path)
    print(f"Saved {len(records)} records to {path} in {time.perf_counter() - started:.2f}s")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--oracle_rollout_policy", choices=["fifo", "lookahead"], default="fifo")
    parser.add_argument("--output_dir", default="data/ranker_dataset/")
    args = parser.parse_args()

    generate_dataset(
        size=args.size,
        split=args.split,
        top_k=args.top_k,
        max_instances=args.max_instances,
        oracle_rollout_policy=args.oracle_rollout_policy,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
