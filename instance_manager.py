"""Fixed dataset manager for HGCR-PPO Stage A experiments."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, List

from src.instances.instance_generator import generate_instance


FIXED_ROOT = Path("data/instances/fixed")
SPLIT_COUNTS = {"train": 200, "val": 30, "test": 50}
SIZE_SEED_BASE = {"small": 110000, "medium": 210000, "large": 310000}
SPLIT_SEED_OFFSET = {"train": 0, "val": 30000, "test": 60000}
SIZES = ["small", "medium", "large"]
SPLITS = ["train", "val", "test"]


def fixed_instance_path(size: str, split: str, index: int, seed: int) -> Path:
    return FIXED_ROOT / split / f"{size}_{split}_{index:04d}_seed_{seed}.pkl"


def _annotate_instance(instance, size: str, split: str, index: int, seed: int):
    instance.instance_id = f"{size}_{split}_{index:04d}_seed_{seed}"
    instance.size = size
    instance.seed = seed
    instance.split = split
    instance.process_type = {job.job_id: job.process_type for job in instance.jobs}
    instance.release_time = {job.job_id: job.release_time for job in instance.jobs}
    instance.candidate_machines = {job.job_id: list(job.candidate_machines) for job in instance.jobs}
    instance.max_split_num = {job.job_id: job.max_split_num for job in instance.jobs}
    return instance


def generate_fixed_instances(
    sizes: Iterable[str] = SIZES,
    splits: Iterable[str] = SPLITS,
    regenerate: bool = False,
) -> None:
    for split in splits:
        (FIXED_ROOT / split).mkdir(parents=True, exist_ok=True)
        for size in sizes:
            for index in range(SPLIT_COUNTS[split]):
                seed = SIZE_SEED_BASE[size] + SPLIT_SEED_OFFSET[split] + index
                path = fixed_instance_path(size, split, index, seed)
                if path.exists() and not regenerate:
                    continue
                instance = _annotate_instance(generate_instance(size=size, seed=seed), size, split, index, seed)
                with path.open("wb") as f:
                    pickle.dump(instance, f)


def fixed_dataset_exists(size: str, split: str) -> bool:
    return len(list((FIXED_ROOT / split).glob(f"{size}_{split}_*_seed_*.pkl"))) >= SPLIT_COUNTS[split]


def fixed_dataset_count(size: str, split: str) -> int:
    return len(list((FIXED_ROOT / split).glob(f"{size}_{split}_*_seed_*.pkl")))


def ensure_fixed_dataset(sizes: Iterable[str] = SIZES, splits: Iterable[str] = SPLITS) -> None:
    missing = [(size, split) for size in sizes for split in splits if not fixed_dataset_exists(size, split)]
    if missing:
        generate_fixed_instances({size for size, _ in missing}, {split for _, split in missing}, regenerate=False)


def load_fixed_instances(size: str, split: str) -> List:
    paths = sorted((FIXED_ROOT / split).glob(f"{size}_{split}_*_seed_*.pkl"))
    if not paths:
        raise FileNotFoundError(
            f"No fixed {size}/{split} instances found. Run: python instance_manager.py --generate"
        )
    instances = []
    for path in paths:
        with path.open("rb") as f:
            instance = pickle.load(f)
        if not hasattr(instance, "instance_id"):
            instance.instance_id = path.stem
        if not hasattr(instance, "size"):
            instance.size = size
        if not hasattr(instance, "split"):
            instance.split = split
        if not hasattr(instance, "seed"):
            try:
                instance.seed = int(path.stem.rsplit("_seed_", 1)[1])
            except (IndexError, ValueError):
                instance.seed = None
        if not hasattr(instance, "process_type"):
            instance.process_type = {job.job_id: job.process_type for job in instance.jobs}
        if not hasattr(instance, "release_time"):
            instance.release_time = {job.job_id: job.release_time for job in instance.jobs}
        if not hasattr(instance, "candidate_machines"):
            instance.candidate_machines = {job.job_id: list(job.candidate_machines) for job in instance.jobs}
        if not hasattr(instance, "max_split_num"):
            instance.max_split_num = {job.job_id: job.max_split_num for job in instance.jobs}
        instances.append(instance)
    return instances


def summarize_fixed_instances() -> None:
    print("split,size,count")
    for split in SPLITS:
        for size in SIZES:
            count = len(list((FIXED_ROOT / split).glob(f"{size}_{split}_*_seed_*.pkl")))
            print(f"{split},{size},{count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    args = parser.parse_args()
    if args.generate:
        generate_fixed_instances(args.sizes, args.splits, regenerate=args.regenerate)
    if args.summary:
        summarize_fixed_instances()
    if not args.generate and not args.summary:
        parser.print_help()


if __name__ == "__main__":
    main()
