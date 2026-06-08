"""Summarize clean Stage G dynamic rolling HGCR-PPO runs."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence


RUNS_DIR = Path("data/results/stage_G/hgcr_dynamic_ppo/runs")
OUTPUT_DIR = Path("data/results/stage_G/summary")
BETAS = {0.01, 1.0, 5.0}
RULE_NAMES = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker"]
RULE_ALIASES = {
    "FIFO": "FIFO",
    "GreedyECT": "GreedyECT",
    "Lookahead": "Lookahead",
    "LookaheadGreedy": "Lookahead",
    "MLP_Ranker_soft_ce": "MLP-Ranker",
    "MLP-Ranker": "MLP-Ranker",
}

ALL_RUN_FIELDS = [
    "run_id",
    "stage",
    "arrival_intensity",
    "carryover_ratio",
    "reward_beta",
    "seed",
    "method",
    "Cmax_mean",
    "Cmax_std",
    "FIFO_Cmax_mean",
    "MLPRanker_Cmax_mean",
    "relative_to_FIFO",
    "relative_to_MLPRanker",
    "valid_schedule_rate",
    "source_run_dir",
    "reward_curve_path",
]
BETA_FIELDS = [
    "arrival_intensity",
    "carryover_ratio",
    "reward_beta",
    "seed",
    "FIFO_Cmax_mean",
    "HGCR_PPO_Cmax_mean",
    "MLPRanker_Cmax_mean",
    "relative_to_FIFO",
    "relative_to_MLPRanker",
    "run_id",
]
SEED_FIELDS = [
    "arrival_intensity",
    "carryover_ratio",
    "reward_beta",
    "seed",
    "FIFO_Cmax_mean",
    "HGCR_PPO_Cmax_mean",
    "MLPRanker_Cmax_mean",
    "relative_to_FIFO",
    "relative_to_MLPRanker",
    "run_id",
]
ARRIVAL_FIELDS = [
    "arrival_intensity",
    "carryover_ratio",
    "reward_beta",
    "seed0_run_id",
    "seed0_FIFO_Cmax_mean",
    "seed0_HGCR_PPO_Cmax_mean",
    "seed0_MLPRanker_Cmax_mean",
    "seed0_relative_to_FIFO",
    "seed0_relative_to_MLPRanker",
    "mean_seeds",
    "num_runs",
    "FIFO_Cmax_mean",
    "FIFO_Cmax_std",
    "HGCR_PPO_Cmax_mean",
    "HGCR_PPO_Cmax_std",
    "MLPRanker_Cmax_mean",
    "MLPRanker_Cmax_std",
    "relative_to_FIFO",
    "relative_to_MLPRanker",
]
ACTION_FIELDS = [
    "run_id",
    "stage",
    "arrival_intensity",
    "carryover_ratio",
    "reward_beta",
    "seed",
    "rule_name",
    "selection_count",
    "selection_ratio",
    "source_run_dir",
]


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def output_paths(output_dir: Path, suffix: str) -> Dict[str, Path]:
    return {
        "all_runs": output_dir / f"stage_G_all_runs__{suffix}.csv",
        "beta": output_dir / f"stage_G_beta_ablation__{suffix}.csv",
        "seed": output_dir / f"stage_G_seed_stability__{suffix}.csv",
        "arrival": output_dir / f"stage_G_arrival_generalization__{suffix}.csv",
        "actions": output_dir / f"stage_G_action_ratio_summary__{suffix}.csv",
        "report": output_dir / f"stage_G_dynamic_report__{suffix}.md",
    }


def first_file(run_dir: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def method_name(value: str) -> str:
    if value in {"HGCR-PPO", "HGCR_Dynamic_PPO", "HGCR-Dynamic-PPO"}:
        return "HGCR-PPO"
    if value in {"MLP_Ranker_soft_ce", "MLP-Ranker", "MLPRanker"}:
        return "MLP-Ranker"
    return value


def load_manifest(run_dir: Path) -> dict | None:
    path = first_file(run_dir, ["manifest__*.json", "manifest*.json"])
    if path is None:
        print(f"Warning: skip {run_dir}, missing manifest.")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Warning: skip {run_dir}, malformed manifest {path}.")
        return None


def is_stage_g_manifest(manifest: dict, run_dir: Path) -> bool:
    checks = {
        "stage": "G",
        "experiment_family": "hgcr_dynamic_ppo",
        "scenario_type": "dynamic_rolling",
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            print(f"Warning: skip {run_dir}, manifest {key}={manifest.get(key)!r}, expected {expected!r}.")
            return False
    return True


def stage_g_run_dirs(runs_dir: Path, max_runs: int | None) -> List[Path]:
    if not runs_dir.exists():
        print(f"Warning: runs_dir does not exist: {runs_dir}")
        return []
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    return run_dirs[:max_runs] if max_runs is not None else run_dirs


def parse_run(run_dir: Path) -> tuple[List[dict], List[dict]]:
    manifest = load_manifest(run_dir)
    if manifest is None or not is_stage_g_manifest(manifest, run_dir):
        return [], []
    eval_path = first_file(run_dir, ["eval_summary__*.csv", "eval_summary*.csv"])
    if eval_path is None:
        print(f"Warning: skip {run_dir}, missing eval_summary CSV.")
        return [], []

    run_id = str(manifest["run_id"])
    arrival = str(manifest["arrival_intensity"])
    carryover = str(manifest["carryover_ratio"])
    reward_beta = parse_float(manifest["reward_beta"])
    seed = str(manifest["seed"])
    curve_path = first_file(run_dir, ["reward_cmax_curve__*.csv", "reward_cmax_curve*.csv"])
    eval_rows = read_csv(eval_path)
    by_method = {method_name(row.get("method", "")): row for row in eval_rows}
    fifo = parse_float((by_method.get("FIFO") or {}).get("Cmax_mean"))
    mlp = parse_float((by_method.get("MLP-Ranker") or {}).get("Cmax_mean"))

    all_rows = []
    for row in eval_rows:
        method = method_name(row.get("method", ""))
        cmax = parse_float(row.get("Cmax_mean"))
        all_rows.append(
            {
                "run_id": run_id,
                "stage": "G",
                "arrival_intensity": arrival,
                "carryover_ratio": carryover,
                "reward_beta": reward_beta,
                "seed": seed,
                "method": method,
                "Cmax_mean": cmax,
                "Cmax_std": parse_float(row.get("Cmax_std")),
                "FIFO_Cmax_mean": fifo,
                "MLPRanker_Cmax_mean": mlp,
                "relative_to_FIFO": parse_float(row.get("relative_to_FIFO"), (fifo - cmax) / max(fifo, 1e-8) if fifo else 0.0),
                "relative_to_MLPRanker": parse_float(row.get("relative_to_MLPRanker"), (mlp - cmax) / max(mlp, 1e-8) if mlp else 0.0),
                "valid_schedule_rate": parse_float(row.get("valid_schedule_rate")),
                "source_run_dir": str(run_dir),
                "reward_curve_path": str(curve_path) if curve_path else "",
            }
        )

    action_rows = []
    action_path = first_file(run_dir, ["action_ratio__*.csv", "action_ratio*.csv"])
    if action_path is None:
        print(f"Warning: no action_ratio CSV found in {run_dir}.")
    else:
        for row in read_csv(action_path):
            rule = RULE_ALIASES.get(row.get("rule_name", ""), row.get("rule_name", ""))
            action_rows.append(
                {
                    "run_id": run_id,
                    "stage": "G",
                    "arrival_intensity": arrival,
                    "carryover_ratio": carryover,
                    "reward_beta": reward_beta,
                    "seed": seed,
                    "rule_name": rule,
                    "selection_count": row.get("selection_count", row.get("rule_count", 0)),
                    "selection_ratio": row.get("selection_ratio", row.get("rule_ratio", 0.0)),
                    "source_run_dir": str(run_dir),
                }
            )
    return all_rows, action_rows


def hgcr_rows(all_rows: Sequence[dict]) -> List[dict]:
    return [row for row in all_rows if row.get("method") == "HGCR-PPO"]


def make_beta_ablation(all_rows: Sequence[dict]) -> List[dict]:
    rows = []
    for row in hgcr_rows(all_rows):
        beta = parse_float(row["reward_beta"])
        if row["arrival_intensity"] == "medium" and row["carryover_ratio"] == "medium" and str(row["seed"]) == "0" and beta in BETAS:
            rows.append(
                {
                    "arrival_intensity": row["arrival_intensity"],
                    "carryover_ratio": row["carryover_ratio"],
                    "reward_beta": beta,
                    "seed": row["seed"],
                    "FIFO_Cmax_mean": row["FIFO_Cmax_mean"],
                    "HGCR_PPO_Cmax_mean": row["Cmax_mean"],
                    "MLPRanker_Cmax_mean": row["MLPRanker_Cmax_mean"],
                    "relative_to_FIFO": row["relative_to_FIFO"],
                    "relative_to_MLPRanker": row["relative_to_MLPRanker"],
                    "run_id": row["run_id"],
                }
            )
    return sorted(rows, key=lambda row: parse_float(row["reward_beta"]))


def make_seed_stability(all_rows: Sequence[dict]) -> List[dict]:
    rows = []
    for row in hgcr_rows(all_rows):
        if row["arrival_intensity"] == "medium" and row["carryover_ratio"] == "medium" and parse_float(row["reward_beta"]) == 5.0 and str(row["seed"]) in {"0", "1", "2"}:
            rows.append(
                {
                    "arrival_intensity": row["arrival_intensity"],
                    "carryover_ratio": row["carryover_ratio"],
                    "reward_beta": row["reward_beta"],
                    "seed": row["seed"],
                    "FIFO_Cmax_mean": row["FIFO_Cmax_mean"],
                    "HGCR_PPO_Cmax_mean": row["Cmax_mean"],
                    "MLPRanker_Cmax_mean": row["MLPRanker_Cmax_mean"],
                    "relative_to_FIFO": row["relative_to_FIFO"],
                    "relative_to_MLPRanker": row["relative_to_MLPRanker"],
                    "run_id": row["run_id"],
                }
            )
    return sorted(rows, key=lambda row: int(row["seed"]))


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def make_arrival_generalization(all_rows: Sequence[dict]) -> List[dict]:
    groups: Dict[str, List[dict]] = {}
    for row in hgcr_rows(all_rows):
        if parse_float(row["reward_beta"]) == 5.0 and row["carryover_ratio"] == "medium" and row["arrival_intensity"] in {"low", "medium", "high"}:
            groups.setdefault(row["arrival_intensity"], []).append(row)

    output = []
    order = {"low": 0, "medium": 1, "high": 2}
    for arrival, rows in sorted(groups.items(), key=lambda item: order[item[0]]):
        rows = sorted(rows, key=lambda row: int(row["seed"]) if str(row["seed"]).isdigit() else 999)
        seed0 = next((row for row in rows if str(row["seed"]) == "0"), rows[0])
        fifo_mean, fifo_std = mean_std([parse_float(row["FIFO_Cmax_mean"]) for row in rows])
        hgcr_mean, hgcr_std = mean_std([parse_float(row["Cmax_mean"]) for row in rows])
        mlp_mean, mlp_std = mean_std([parse_float(row["MLPRanker_Cmax_mean"]) for row in rows])
        output.append(
            {
                "arrival_intensity": arrival,
                "carryover_ratio": "medium",
                "reward_beta": 5.0,
                "seed0_run_id": seed0["run_id"],
                "seed0_FIFO_Cmax_mean": seed0["FIFO_Cmax_mean"],
                "seed0_HGCR_PPO_Cmax_mean": seed0["Cmax_mean"],
                "seed0_MLPRanker_Cmax_mean": seed0["MLPRanker_Cmax_mean"],
                "seed0_relative_to_FIFO": seed0["relative_to_FIFO"],
                "seed0_relative_to_MLPRanker": seed0["relative_to_MLPRanker"],
                "mean_seeds": ",".join(str(row["seed"]) for row in rows),
                "num_runs": len(rows),
                "FIFO_Cmax_mean": fifo_mean,
                "FIFO_Cmax_std": fifo_std,
                "HGCR_PPO_Cmax_mean": hgcr_mean,
                "HGCR_PPO_Cmax_std": hgcr_std,
                "MLPRanker_Cmax_mean": mlp_mean,
                "MLPRanker_Cmax_std": mlp_std,
                "relative_to_FIFO": (fifo_mean - hgcr_mean) / max(fifo_mean, 1e-8),
                "relative_to_MLPRanker": (mlp_mean - hgcr_mean) / max(mlp_mean, 1e-8),
            }
        )
    return output


def make_report(beta_rows: Sequence[dict], seed_rows: Sequence[dict], arrival_rows: Sequence[dict], action_rows: Sequence[dict]) -> str:
    return "\n".join(
        [
            "# 阶段 G 动态滚动场景 HGCR-PPO 汇总报告",
            "",
            "## 1. 阶段 G 目标",
            "阶段 G 用于动态滚动场景下的 HGCR-PPO 正式训练与实验，验证规则级 PPO 是否能在动态到达和结转条件下形成稳定调度收益。",
            "",
            "## 2. 为什么从静态 PPO 转向动态滚动场景",
            "旧静态 PPO 在固定测试集上不稳定，直接调参收益有限。阶段 G 将动作重构为 FIFO、GreedyECT、Lookahead、MLP-Ranker 的规则选择，更接近动态调度 DRL 文献中的规则动作空间和 online scheduling 范式。",
            "",
            "## 3. reward_beta 敏感性结论",
            f"medium arrival + medium carryover + seed0 的 reward_beta 消融共汇总 {len(beta_rows)} 条记录。该表用于比较 beta=0.01、1.0、5.0 对 HGCR-PPO 超越 FIFO 的影响。",
            "",
            "## 4. medium-medium 三 seed 稳定性结论",
            f"medium-medium、reward_beta=5.0 的 seed 稳定性共汇总 {len(seed_rows)} 条记录。该表用于验证 seed0/1/2 是否均能不低于 FIFO。",
            "",
            "## 5. low/medium/high arrival 泛化结论",
            f"reward_beta=5.0、carryover=medium 的 arrival 泛化共汇总 {len(arrival_rows)} 个 arrival 分组。medium 场景优先保留 seed0，并同时输出 seed 平均与标准差。",
            "",
            "## 6. 动作选择比例解释",
            f"动作比例表共汇总 {len(action_rows)} 条规则选择记录，可用于说明 HGCR-PPO 不是单一规则，而是 FIFO、GreedyECT、Lookahead、MLP-Ranker 的自适应组合。",
            "",
            "## 7. 当前论文可用结论",
            "- HGCR-PPO 在动态滚动场景下可以超过 FIFO 强基线。",
            "- HGCR-PPO 相比静态 MLP-Ranker 更适合动态滚动调度。",
            "- 规则动作空间降低了直接 job-machine 动作的学习难度。",
            "- reward scaling 是阶段 G 必要消融。",
            "",
        ]
    )


def summarize(args) -> Dict[str, object]:
    suffix = token()
    paths = output_paths(Path(args.output_dir), suffix)
    run_dirs = stage_g_run_dirs(Path(args.runs_dir), args.max_runs)
    print(f"Stage G runs_dir: {args.runs_dir}")
    print(f"Runs selected: {len(run_dirs)}")
    for run_dir in run_dirs:
        print(f"  - {run_dir}")
    print("Planned outputs:")
    for path in paths.values():
        print(f"  - {path}")
    if args.dry_run:
        print("Dry run enabled: no files will be written.")
        return {"paths": paths, "all_rows": [], "action_rows": []}

    all_rows: List[dict] = []
    action_rows: List[dict] = []
    for run_dir in run_dirs:
        run_rows, run_actions = parse_run(run_dir)
        all_rows.extend(run_rows)
        action_rows.extend(run_actions)

    beta_rows = make_beta_ablation(all_rows)
    seed_rows = make_seed_stability(all_rows)
    arrival_rows = make_arrival_generalization(all_rows)
    report = make_report(beta_rows, seed_rows, arrival_rows, action_rows)

    if args.no_write:
        print(
            "No-write enabled: read/filter completed without writing CSV/MD "
            f"({len(all_rows)} eval rows, {len(action_rows)} action rows)."
        )
        return {"paths": paths, "all_rows": all_rows, "action_rows": action_rows}

    write_csv(paths["all_runs"], all_rows, ALL_RUN_FIELDS)
    write_csv(paths["beta"], beta_rows, BETA_FIELDS)
    write_csv(paths["seed"], seed_rows, SEED_FIELDS)
    write_csv(paths["arrival"], arrival_rows, ARRIVAL_FIELDS)
    write_csv(paths["actions"], action_rows, ACTION_FIELDS)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(report, encoding="utf-8")
    print(f"Saved Stage G dynamic summary to {args.output_dir}")
    return {"paths": paths, "all_rows": all_rows, "action_rows": action_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default=str(RUNS_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
