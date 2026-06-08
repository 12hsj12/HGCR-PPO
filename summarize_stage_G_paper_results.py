"""Build paper-oriented Stage G result tables."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Sequence


HGCR_RUNS_DIR = Path("data/results/stage_G/hgcr_dynamic_ppo/runs")
BASELINE_DIR = Path("data/results/stage_G/baseline_eval/runs")
OUTPUT_DIR = Path("data/results/stage_G/paper_results")
METHOD_ORDER = ["Random", "SPT", "LPT", "MinLoad", "GreedyECT", "Lookahead", "FIFO", "MLP-Ranker", "HGCR-PPO"]
RULES = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker"]


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def latest_paths(output_dir: Path, suffix: str) -> Dict[str, Path]:
    return {
        "detail": output_dir / f"stage_G_method_comparison_detail__{suffix}.csv",
        "summary": output_dir / f"stage_G_method_comparison_summary__{suffix}.csv",
        "wtl": output_dir / f"stage_G_win_tie_loss__{suffix}.csv",
        "rank": output_dir / f"stage_G_rank_summary__{suffix}.csv",
        "heatmap": output_dir / f"stage_G_scenario_heatmap__{suffix}.csv",
        "action_perf": output_dir / f"stage_G_action_performance_summary__{suffix}.csv",
        "report": output_dir / f"stage_G_paper_report__{suffix}.md",
    }


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_baseline_detail(root: Path) -> List[dict]:
    rows: List[dict] = []
    if not root.exists():
        print(f"Warning: baseline eval root does not exist: {root}")
        return rows
    for path in sorted(root.glob("*/baseline_eval_detail__*.csv")):
        rows.extend(read_csv(path))
    return rows


def collect_stage_g_actions(root: Path) -> List[dict]:
    rows = []
    if not root.exists():
        return rows
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = next(iter(sorted(run_dir.glob("manifest*.json"))), None)
        action_path = next(iter(sorted(run_dir.glob("action_ratio*.csv"))), None)
        eval_path = next(iter(sorted(run_dir.glob("eval_summary*.csv"))), None)
        if manifest_path is None or action_path is None or eval_path is None:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("stage") != "G":
            continue
        eval_rows = read_csv(eval_path)
        by_method = {row["method"]: row for row in eval_rows}
        hgcr = by_method.get("HGCR-PPO", {})
        fifo = by_method.get("FIFO", {})
        mlp = by_method.get("MLP_Ranker_soft_ce") or by_method.get("MLP-Ranker") or {}
        ratios = {rule: 0.0 for rule in RULES}
        for row in read_csv(action_path):
            rule = row.get("rule_name", "")
            if rule == "MLP_Ranker_soft_ce":
                rule = "MLP-Ranker"
            if rule in ratios:
                ratios[rule] = fnum(row.get("selection_ratio"))
        rows.append(
            {
                "arrival_intensity": manifest.get("arrival_intensity", ""),
                "carryover_ratio": manifest.get("carryover_ratio", ""),
                "seed": manifest.get("seed", ""),
                "FIFO_ratio": ratios["FIFO"],
                "GreedyECT_ratio": ratios["GreedyECT"],
                "Lookahead_ratio": ratios["Lookahead"],
                "MLPRanker_ratio": ratios["MLP-Ranker"],
                "HGCR_Cmax": fnum(hgcr.get("Cmax_mean")),
                "FIFO_Cmax": fnum(fifo.get("Cmax_mean")),
                "MLPRanker_Cmax": fnum(mlp.get("Cmax_mean")),
                "HGCR_relative_to_FIFO": fnum(hgcr.get("relative_to_FIFO")),
                "HGCR_relative_to_MLP": fnum(hgcr.get("relative_to_MLPRanker")),
            }
        )
    return rows


def method_summary(detail: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in detail:
        key = (row["arrival_intensity"], row["carryover_ratio"], row["seed"], row["method"])
        groups.setdefault(key, []).append(row)
    out = []
    for (arrival, carryover, seed, method), vals in sorted(groups.items()):
        cmax = [fnum(row["Cmax"]) for row in vals]
        out.append(
            {
                "arrival_intensity": arrival,
                "carryover_ratio": carryover,
                "seed": seed,
                "method": method,
                "Cmax_mean": mean(cmax),
                "Cmax_std": pstdev(cmax) if len(cmax) > 1 else 0.0,
                "Cmax_min": min(cmax),
                "Cmax_max": max(cmax),
                "instance_count": len(vals),
            }
        )
    return out


def win_tie_loss(detail: Sequence[dict], tolerance: float = 1e-6) -> List[dict]:
    by_instance: Dict[tuple, Dict[str, float]] = {}
    meta: Dict[tuple, dict] = {}
    for row in detail:
        key = (row["scenario_run_id"], row["instance_id"])
        by_instance.setdefault(key, {})[row["method"]] = fnum(row["Cmax"])
        meta[key] = row
    out = []
    baselines = [m for m in METHOD_ORDER if m != "HGCR-PPO"]
    for method in baselines:
        gaps = []
        wins = ties = losses = 0
        for values in by_instance.values():
            if "HGCR-PPO" not in values or method not in values:
                continue
            gap = values[method] - values["HGCR-PPO"]
            gaps.append(gap)
            if gap > tolerance:
                wins += 1
            elif gap < -tolerance:
                losses += 1
            else:
                ties += 1
        total = max(1, wins + ties + losses)
        out.append(
            {
                "baseline_method": method,
                "win_count": wins,
                "tie_count": ties,
                "loss_count": losses,
                "win_rate": wins / total,
                "tie_rate": ties / total,
                "loss_rate": losses / total,
                "mean_gap": mean(gaps) if gaps else 0.0,
                "median_gap": median(gaps) if gaps else 0.0,
            }
        )
    return out


def rank_summary(detail: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in detail:
        groups.setdefault((row["scenario_run_id"], row["instance_id"]), []).append(row)
    ranks_by_method: Dict[str, List[int]] = {}
    for rows in groups.values():
        ranked = sorted(rows, key=lambda row: (fnum(row["Cmax"]), METHOD_ORDER.index(row["method"]) if row["method"] in METHOD_ORDER else 99))
        for idx, row in enumerate(ranked, start=1):
            ranks_by_method.setdefault(row["method"], []).append(idx)
    out = []
    for method, ranks in sorted(ranks_by_method.items(), key=lambda item: METHOD_ORDER.index(item[0]) if item[0] in METHOD_ORDER else 99):
        out.append(
            {
                "method": method,
                "mean_rank": mean(ranks),
                "median_rank": median(ranks),
                "rank1_count": sum(1 for r in ranks if r == 1),
                "top2_count": sum(1 for r in ranks if r <= 2),
                "top3_count": sum(1 for r in ranks if r <= 3),
                "instance_count": len(ranks),
            }
        )
    return out


def scenario_heatmap(summary_rows: Sequence[dict], rank_rows: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, Dict[str, List[float]]] = {}
    for row in summary_rows:
        key = (row["arrival_intensity"], row["carryover_ratio"])
        groups.setdefault(key, {}).setdefault(row["method"], []).append(fnum(row["Cmax_mean"]))
    hgcr_rank = next((fnum(row["mean_rank"]) for row in rank_rows if row["method"] == "HGCR-PPO"), 0.0)
    out = []
    for (arrival, carryover), methods in sorted(groups.items()):
        hgcr = mean(methods.get("HGCR-PPO", [0.0]))
        fifo = mean(methods.get("FIFO", [0.0]))
        mlp = mean(methods.get("MLP-Ranker", [0.0]))
        out.append(
            {
                "arrival_intensity": arrival,
                "carryover_ratio": carryover,
                "HGCR_PPO_Cmax_mean": hgcr,
                "FIFO_Cmax_mean": fifo,
                "MLPRanker_Cmax_mean": mlp,
                "HGCR_improvement_over_FIFO": (fifo - hgcr) / max(fifo, 1e-8),
                "HGCR_improvement_over_MLP": (mlp - hgcr) / max(mlp, 1e-8),
                "HGCR_mean_rank": hgcr_rank,
            }
        )
    return out


def report_text(detail, summary, wtl, rank, heatmap, action_perf) -> str:
    return "\n".join(
        [
            "# 阶段 G 论文级结果汇总",
            "",
            "## 1. 阶段 G 目标",
            "阶段 G 验证 HGCR-PPO 在动态滚动场景下相对于多种调度规则和学习基线的有效性。",
            "",
            "## 2. 为什么需要多基线和 per-instance 对比",
            "HGCR-PPO 相对 FIFO 的均值提升可能较小，仅看均值柱状图不够充分。多基线、实例级胜负和排名统计能更清楚展示动态规则组合的稳健性。",
            "",
            "## 3. 多算法整体结果",
            f"当前合并 {len(detail)} 条 per-instance 记录，形成 {len(summary)} 条方法-场景汇总。",
            "",
            "## 4. 胜负统计结果",
            f"win/tie/loss 表包含 {len(wtl)} 个 HGCR-PPO 对比基线。",
            "",
            "## 5. 动态扰动热力图结果",
            f"heatmap 表包含 {len(heatmap)} 个 arrival × carryover 场景。",
            "",
            "## 6. 动作选择与性能关系解释",
            f"动作-性能表包含 {len(action_perf)} 条 run 级记录，可用于解释规则选择比例和 Cmax 改进的关系。",
            "",
            "## 7. 甘特图实例建议",
            "建议选择 HGCR-PPO 相比 FIFO gap 最大的实例，展示 FIFO 与 HGCR-PPO 的产线占用差异、Cmax 和利用率。",
            "",
            "## 8. 论文图表建议",
            "主文优先使用多算法对比、per-instance 箱线图、win/tie/loss、rank summary、动作-性能 panel；reward scaling 和训练曲线作为补充图。",
            "",
        ]
    )


def run(args):
    suffix = token()
    paths = latest_paths(Path(args.output_dir), suffix)
    print(f"Baseline input: {args.baseline_dir}")
    print(f"HGCR run input: {args.hgcr_runs_dir}")
    print("Planned outputs:")
    for path in paths.values():
        print(f"  - {path}")
    if args.dry_run:
        print("Dry run enabled: no files will be written.")
        return paths
    detail = collect_baseline_detail(Path(args.baseline_dir))
    summary = method_summary(detail)
    wtl = win_tie_loss(detail)
    rank = rank_summary(detail)
    heatmap = scenario_heatmap(summary, rank)
    action_perf = collect_stage_g_actions(Path(args.hgcr_runs_dir))
    if args.no_write:
        print(f"No-write enabled: read {len(detail)} detail rows, no files written.")
        return paths
    write_csv(paths["detail"], detail, list(detail[0].keys()) if detail else [])
    write_csv(paths["summary"], summary, ["arrival_intensity", "carryover_ratio", "seed", "method", "Cmax_mean", "Cmax_std", "Cmax_min", "Cmax_max", "instance_count"])
    write_csv(paths["wtl"], wtl, ["baseline_method", "win_count", "tie_count", "loss_count", "win_rate", "tie_rate", "loss_rate", "mean_gap", "median_gap"])
    write_csv(paths["rank"], rank, ["method", "mean_rank", "median_rank", "rank1_count", "top2_count", "top3_count", "instance_count"])
    write_csv(paths["heatmap"], heatmap, ["arrival_intensity", "carryover_ratio", "HGCR_PPO_Cmax_mean", "FIFO_Cmax_mean", "MLPRanker_Cmax_mean", "HGCR_improvement_over_FIFO", "HGCR_improvement_over_MLP", "HGCR_mean_rank"])
    write_csv(paths["action_perf"], action_perf, ["arrival_intensity", "carryover_ratio", "seed", "FIFO_ratio", "GreedyECT_ratio", "Lookahead_ratio", "MLPRanker_ratio", "HGCR_Cmax", "FIFO_Cmax", "MLPRanker_Cmax", "HGCR_relative_to_FIFO", "HGCR_relative_to_MLP"])
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(report_text(detail, summary, wtl, rank, heatmap, action_perf), encoding="utf-8")
    print(f"Saved Stage G paper results to {args.output_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hgcr_runs_dir", default=str(HGCR_RUNS_DIR))
    parser.add_argument("--baseline_dir", default=str(BASELINE_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
