"""Fixed instance dataset generation and summary utilities."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from src.instances.instance_generator import generate_instance


DATASET_DIR = Path("data/instances")
DATASET_SPECS = {
    "train_small": ("small", "train", 30, 1001),
    "test_small": ("small", "test", 10, 2001),
    "train_medium": ("medium", "train", 20, 3001),
    "test_medium": ("medium", "test", 10, 4001),
    "train_large": ("large", "train", 10, 5001),
    "test_large": ("large", "test", 10, 6001),
}


def instance_path(size: str, split: str, seed: int) -> Path:
    return DATASET_DIR / f"{size}_{split}_seed_{seed}.pkl"


def generate_fixed_datasets(regenerate: bool = False) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for _, (size, split, count, start_seed) in DATASET_SPECS.items():
        for offset in range(count):
            seed = start_seed + offset
            path = instance_path(size, split, seed)
            if path.exists() and not regenerate:
                continue
            instance = generate_instance(size=size, seed=seed)
            with path.open("wb") as f:
                pickle.dump(instance, f)
            print(f"saved {path}")


def load_instance(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_dataset(size: str, split: str) -> List:
    paths = sorted(DATASET_DIR.glob(f"{size}_{split}_seed_*.pkl"))
    if not paths:
        raise FileNotFoundError(
            f"No fixed instances found for {size}/{split}. Run: python dataset_manager.py --generate_all"
        )
    return [load_instance(path) for path in paths]


def iter_all_instances() -> Iterable[tuple[str, str, Path, object]]:
    for path in sorted(DATASET_DIR.glob("*_*_seed_*.pkl")):
        parts = path.stem.split("_")
        if len(parts) >= 4:
            size, split = parts[0], parts[1]
            yield size, split, path, load_instance(path)


def dataset_summary() -> List[Dict[str, float]]:
    grouped: Dict[tuple[str, str], List] = {}
    for size, split, _, instance in iter_all_instances():
        grouped.setdefault((size, split), []).append(instance)

    rows = []
    for (size, split), instances in sorted(grouped.items()):
        job_counts = [len(inst.jobs) for inst in instances]
        machine_counts = [len(inst.machines) for inst in instances]
        period_counts = [inst.num_periods for inst in instances]
        split_ratios = [
            sum(1 for job in inst.jobs if job.max_split_num > 1) / len(inst.jobs)
            for inst in instances
        ]
        candidate_counts = [
            mean(len(job.candidate_machines) for job in inst.jobs) for inst in instances
        ]
        pt_ranges = []
        for inst in instances:
            vals = [time for times in inst.processing_time.values() for time in times.values()]
            pt_ranges.append(max(vals) - min(vals))
        rows.append(
            {
                "size": size,
                "split": split,
                "count": len(instances),
                "avg_jobs": mean(job_counts),
                "avg_machines": mean(machine_counts),
                "avg_periods": mean(period_counts),
                "avg_splittable_ratio": mean(split_ratios),
                "avg_candidate_machines": mean(candidate_counts),
                "avg_processing_time_range": mean(pt_ranges),
            }
        )
    return rows


def print_summary() -> None:
    rows = dataset_summary()
    if not rows:
        print("No fixed instances found. Run: python dataset_manager.py --generate_all")
        return
    header = (
        "size     split   count avg_jobs avg_machines avg_periods "
        "avg_split_ratio avg_candidates avg_pt_range"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['size']:<8} {row['split']:<7} {row['count']:<5.0f} "
            f"{row['avg_jobs']:<8.2f} {row['avg_machines']:<12.2f} "
            f"{row['avg_periods']:<11.2f} {row['avg_splittable_ratio']:<15.3f} "
            f"{row['avg_candidate_machines']:<14.2f} {row['avg_processing_time_range']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_all", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--regenerate_instances", action="store_true")
    args = parser.parse_args()

    if args.generate_all:
        generate_fixed_datasets(regenerate=args.regenerate_instances)
    if args.summary:
        print_summary()
    if not args.generate_all and not args.summary:
        parser.print_help()


if __name__ == "__main__":
    main()
