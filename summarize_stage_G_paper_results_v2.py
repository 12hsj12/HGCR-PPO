"""Stage G-Plus v2 paper result summarizer.

Reads only Stage G paper result detail CSVs and creates v2 tables for the
final dynamic rolling scheduling figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Sequence


PAPER_DIR = Path("data/results/stage_G/paper_results")
SELECTED_METHODS = ["FIFO", "GreedyECT", "Lookahead", "MinLoad", "MLP-Ranker", "HGCR-PPO"]
SIG_BASELINES = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker", "MinLoad"]


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def paths(output_dir: Path, suffix: str) -> Dict[str, Path]:
    return {
        "summary": output_dir / f"stage_G_method_comparison_summary_v2__{suffix}.csv",
        "arpd": output_dir / f"stage_G_arpd_summary_v2__{suffix}.csv",
        "sig": output_dir / f"stage_G_significance_tests_v2__{suffix}.csv",
        "case": output_dir / f"stage_G_case_curve_detail_v2__{suffix}.csv",
        "scale": output_dir / f"stage_G_scale_summary_v2__{suffix}.csv",
        "case_mapping": output_dir / f"stage_G_case_mapping_v2__{suffix}.csv",
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
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) or math.isinf(value) else value


def collect_detail(root: Path) -> List[dict]:
    if not root.exists():
        print(f"Warning: paper_results directory does not exist: {root}")
        return []
    rows = []
    for path in sorted(root.glob("stage_G_method_comparison_detail__*.csv")):
        rows.extend(read_csv(path))
    return rows


def case_id(row: dict) -> str:
    return row.get("case_id") or f"{row.get('size', 'unknown')}-{row.get('arrival_intensity')}-{row.get('carryover_ratio')}-seed{row.get('seed')}-inst{row.get('instance_id')}"


def size_of(row: dict) -> str:
    return row.get("size") or "unknown"


def method_summary(detail: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in detail:
        key = (size_of(row), row["arrival_intensity"], row["carryover_ratio"], row["reward_beta"], row["method"])
        groups.setdefault(key, []).append(row)
    out = []
    for (size, arrival, carryover, beta, method), rows in sorted(groups.items()):
        cmax = [fnum(r["Cmax"]) for r in rows]
        out.append(
            {
                "size": size,
                "arrival_intensity": arrival,
                "carryover_ratio": carryover,
                "reward_beta": beta,
                "method": method,
                "Cmax_mean": mean(cmax),
                "Cmax_std": pstdev(cmax) if len(cmax) > 1 else 0.0,
                "average_completion_time_mean": mean(fnum(r.get("average_completion_time")) for r in rows),
                "average_waiting_time_mean": mean(fnum(r.get("average_waiting_time")) for r in rows),
                "machine_utilization_mean": mean(fnum(r.get("machine_utilization")) for r in rows),
                "load_balance_std_mean": mean(fnum(r.get("load_balance_std")) for r in rows),
                "runtime_seconds_mean": mean(fnum(r.get("runtime_seconds")) for r in rows),
                "valid_schedule_rate": mean(1.0 if str(r.get("valid_schedule")).lower() == "true" else 0.0 for r in rows),
                "n_instances": len(rows),
            }
        )
    return out


def by_case(detail: Sequence[dict]) -> Dict[str, Dict[str, dict]]:
    cases: Dict[str, Dict[str, dict]] = {}
    for row in detail:
        if row.get("method") in SELECTED_METHODS:
            cases.setdefault(case_id(row), {})[row["method"]] = row
    return cases


def case_curve(detail: Sequence[dict]) -> List[dict]:
    rows = []
    case_labels = {cid: f"C{idx + 1}" for idx, cid in enumerate(sorted(by_case(detail)))}
    for cid, methods in by_case(detail).items():
        if not methods:
            continue
        best = min(fnum(r["Cmax"]) for r in methods.values())
        fifo = fnum(methods.get("FIFO", {}).get("Cmax"))
        mlp = fnum(methods.get("MLP-Ranker", {}).get("Cmax"))
        greedy = fnum(methods.get("GreedyECT", {}).get("Cmax"))
        lookahead = fnum(methods.get("Lookahead", {}).get("Cmax"))
        ranked = sorted(methods.items(), key=lambda item: fnum(item[1]["Cmax"]))
        rank = {m: idx + 1 for idx, (m, _) in enumerate(ranked)}
        for method, row in methods.items():
            cmax = fnum(row["Cmax"])
            rows.append(
                {
                    "case_label": row.get("case_label") or case_labels[cid],
                    "case_id": cid,
                    "size": size_of(row),
                    "arrival_intensity": row["arrival_intensity"],
                    "carryover_ratio": row["carryover_ratio"],
                    "reward_beta": row.get("reward_beta", ""),
                    "seed": row["seed"],
                    "instance_id": row["instance_id"],
                    "method": method,
                    "Cmax": cmax,
                    "Cmax_best_among_selected_methods": best,
                    "improvement_vs_FIFO": (fifo - cmax) / max(fifo, 1e-8) * 100.0 if fifo else 0.0,
                    "improvement_vs_MLPRanker": (mlp - cmax) / max(mlp, 1e-8) * 100.0 if mlp else 0.0,
                    "improvement_vs_GreedyECT": (greedy - cmax) / max(greedy, 1e-8) * 100.0 if greedy else 0.0,
                    "improvement_vs_Lookahead": (lookahead - cmax) / max(lookahead, 1e-8) * 100.0 if lookahead else 0.0,
                    "rank_among_selected_methods": rank[method],
                }
            )
    return rows


def arpd_summary(case_rows: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in case_rows:
        key = (row["size"], row["method"], row["arrival_intensity"], row["carryover_ratio"], row.get("reward_beta", ""))
        groups.setdefault(key, []).append(row)
    out = []
    for (size, method, arrival, carryover, beta), rows in sorted(groups.items()):
        arpd = [(fnum(r["Cmax"]) - fnum(r["Cmax_best_among_selected_methods"])) / max(fnum(r["Cmax_best_among_selected_methods"]), 1e-8) * 100.0 for r in rows]
        ranks = [fnum(r["rank_among_selected_methods"]) for r in rows]
        out.append(
            {
                "size": size,
                "method": method,
                "arrival_intensity": arrival,
                "carryover_ratio": carryover,
                "reward_beta": beta,
                "ARPD_mean": mean(arpd),
                "ARPD_std": pstdev(arpd) if len(arpd) > 1 else 0.0,
                "rank_mean": mean(ranks),
                "rank_std": pstdev(ranks) if len(ranks) > 1 else 0.0,
                "n_instances": len(rows),
            }
        )
    return out


def scale_summary(case_rows: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in case_rows:
        groups.setdefault((row["size"], row["method"]), []).append(row)
        groups.setdefault(("all", row["method"]), []).append(row)
    out = []
    for (size, method), rows in sorted(groups.items()):
        cmax = [fnum(r["Cmax"]) for r in rows]
        arpd = [(fnum(r["Cmax"]) - fnum(r["Cmax_best_among_selected_methods"])) / max(fnum(r["Cmax_best_among_selected_methods"]), 1e-8) * 100.0 for r in rows]
        ranks = [fnum(r["rank_among_selected_methods"]) for r in rows]
        out.append(
            {
                "size": size,
                "method": method,
                "Cmax_mean": mean(cmax),
                "Cmax_std": pstdev(cmax) if len(cmax) > 1 else 0.0,
                "ARPD_mean": mean(arpd),
                "ARPD_std": pstdev(arpd) if len(arpd) > 1 else 0.0,
                "rank_mean": mean(ranks),
                "rank_std": pstdev(ranks) if len(ranks) > 1 else 0.0,
                "n_instances": len(rows),
            }
        )
    return out


def case_mapping(case_rows: Sequence[dict]) -> List[dict]:
    seen = {}
    for row in case_rows:
        seen.setdefault(
            row["case_label"],
            {
                "case_label": row["case_label"],
                "case_id": row["case_id"],
                "size": row["size"],
                "arrival_intensity": row["arrival_intensity"],
                "carryover_ratio": row["carryover_ratio"],
                "seed": row["seed"],
                "instance_id": row["instance_id"],
            },
        )
    return [seen[label] for label in sorted(seen, key=lambda x: int(str(x).lstrip("C") or 0))]


def p_value_paired(diffs: List[float]) -> tuple[str, float | str]:
    if not diffs:
        return "unavailable", ""
    try:
        from scipy.stats import wilcoxon

        return "wilcoxon_signed_rank", float(wilcoxon(diffs, alternative="less").pvalue)
    except Exception:
        pass
    try:
        from scipy.stats import ttest_rel

        return "paired_t_test", float(ttest_rel(diffs, [0.0] * len(diffs), alternative="less").pvalue)
    except Exception:
        return "unavailable", ""


def significance(detail: Sequence[dict]) -> List[dict]:
    cases = by_case(detail)
    out = []
    for baseline in SIG_BASELINES:
        diffs = []
        for methods in cases.values():
            if "HGCR-PPO" in methods and baseline in methods:
                diffs.append(fnum(methods["HGCR-PPO"]["Cmax"]) - fnum(methods[baseline]["Cmax"]))
        test, p = p_value_paired(diffs)
        avg = mean(diffs) if diffs else 0.0
        med = median(diffs) if diffs else 0.0
        significant = (p != "" and float(p) < 0.05)
        out.append(
            {
                "comparison": f"HGCR-PPO_vs_{baseline}",
                "baseline_method": baseline,
                "test_name": test,
                "n_pairs": len(diffs),
                "mean_diff": avg,
                "median_diff": med,
                "p_value": p,
                "significant": significant,
                "effect_direction": "HGCR_better" if avg < 0 else "baseline_better_or_tie",
            }
        )
    return out


def run(args):
    suffix = token()
    out_paths = paths(Path(args.output_dir), suffix)
    detail = collect_detail(Path(args.input_dir))
    print(f"Input detail rows: {len(detail)} from {args.input_dir}")
    for path in out_paths.values():
        print(f"Planned output: {path}")
    if args.dry_run:
        print("Dry run enabled: no files will be written.")
        return out_paths
    cases = case_curve(detail)
    summary = method_summary(detail)
    arpd = arpd_summary(cases)
    scale = scale_summary(cases)
    mapping = case_mapping(cases)
    sig = significance(detail)
    if args.no_write:
        print(f"No-write enabled: prepared {len(summary)} summary rows, {len(cases)} case rows, {len(scale)} scale rows.")
        return out_paths
    write_csv(out_paths["summary"], summary, ["size", "arrival_intensity", "carryover_ratio", "reward_beta", "method", "Cmax_mean", "Cmax_std", "average_completion_time_mean", "average_waiting_time_mean", "machine_utilization_mean", "load_balance_std_mean", "runtime_seconds_mean", "valid_schedule_rate", "n_instances"])
    write_csv(out_paths["arpd"], arpd, ["size", "method", "arrival_intensity", "carryover_ratio", "reward_beta", "ARPD_mean", "ARPD_std", "rank_mean", "rank_std", "n_instances"])
    write_csv(out_paths["sig"], sig, ["comparison", "baseline_method", "test_name", "n_pairs", "mean_diff", "median_diff", "p_value", "significant", "effect_direction"])
    write_csv(out_paths["case"], cases, ["case_label", "case_id", "size", "arrival_intensity", "carryover_ratio", "reward_beta", "seed", "instance_id", "method", "Cmax", "Cmax_best_among_selected_methods", "improvement_vs_FIFO", "improvement_vs_MLPRanker", "improvement_vs_GreedyECT", "improvement_vs_Lookahead", "rank_among_selected_methods"])
    write_csv(out_paths["scale"], scale, ["size", "method", "Cmax_mean", "Cmax_std", "ARPD_mean", "ARPD_std", "rank_mean", "rank_std", "n_instances"])
    write_csv(out_paths["case_mapping"], mapping, ["case_label", "case_id", "size", "arrival_intensity", "carryover_ratio", "seed", "instance_id"])
    print(f"Saved v2 paper results to {args.output_dir}")
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(PAPER_DIR))
    parser.add_argument("--output_dir", default=str(PAPER_DIR))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
