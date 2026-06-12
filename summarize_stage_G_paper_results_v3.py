"""Build clean Stage G paper tables with valid sizes and no external FIFO."""

from __future__ import annotations

import argparse
import csv
import math
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Sequence


PAPER_DIR = Path("data/results/stage_G/paper_results")
VALID_SIZES = ["small", "medium", "large"]
EXTERNAL_METHODS = ["HGCR-PPO", "MLP-Ranker", "GreedyECT", "Lookahead", "MinLoad"]
WEAK_BASELINES = ["Random", "SPT", "LPT"]
SIG_BASELINES = ["MLP-Ranker", "GreedyECT", "Lookahead", "MinLoad"]


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def output_paths(output_dir: Path, suffix: str) -> Dict[str, Path]:
    prefix = "stage_G"
    return {
        "summary": output_dir / f"{prefix}_method_comparison_summary_v3_no_fifo__{suffix}.csv",
        "arpd": output_dir / f"{prefix}_arpd_summary_v3_no_fifo__{suffix}.csv",
        "significance": output_dir / f"{prefix}_significance_tests_v3_no_fifo__{suffix}.csv",
        "case": output_dir / f"{prefix}_case_curve_detail_v3_no_fifo__{suffix}.csv",
        "scale": output_dir / f"{prefix}_scale_summary_v3_no_fifo__{suffix}.csv",
        "audit": output_dir / f"{prefix}_data_audit_v3_no_fifo__{suffix}.csv",
        "mapping": output_dir / f"{prefix}_case_mapping_v3_no_fifo__{suffix}.csv",
    }


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) or math.isinf(number) else number


def has_size_column(path: Path) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            return "size" in (next(csv.reader(f), []) or [])
    except OSError:
        return False


def select_detail_files(args) -> List[Path]:
    explicit = []
    if args.detail_file:
        explicit.append(Path(args.detail_file))
    explicit.extend(Path(path) for path in (args.detail_files or []))
    if explicit:
        missing = [str(path) for path in explicit if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Detail files not found: {missing}")
        return explicit
    root = Path(args.input_dir)
    candidates = [path for path in root.glob("stage_G_method_comparison_detail__*.csv") if has_size_column(path)]
    if not candidates:
        return []
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return [candidates[-1]] if args.use_latest_detail else candidates


def count_by(rows: Sequence[dict], key: str) -> Dict[str, int]:
    counts = Counter((row.get(key) or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def case_id(row: dict) -> str:
    return (
        f"{row.get('size')}-{row.get('arrival_intensity')}-{row.get('carryover_ratio')}"
        f"-b{row.get('reward_beta')}-seed{row.get('seed')}-inst{row.get('instance_id')}"
    )


def audit_and_filter(rows: Sequence[dict], args, source_rows: Sequence[dict]) -> tuple[List[dict], List[dict]]:
    valid_sizes = set(args.valid_sizes)
    excluded = set(args.exclude_methods)
    allowed_methods = set(args.external_methods) | set(args.include_weak_baselines)
    before_size = count_by(rows, "size")
    unknown_rows = [row for row in rows if row.get("size") not in valid_sizes]
    fifo_rows = [row for row in rows if row.get("method") in excluded]
    filtered = []
    for row in rows:
        if row.get("size") not in valid_sizes:
            continue
        if row.get("method") in excluded:
            continue
        if row.get("method") not in allowed_methods:
            continue
        filtered.append(dict(row))

    deduped: Dict[tuple, dict] = {}
    for row in filtered:
        key = (
            row.get("size"),
            row.get("arrival_intensity"),
            row.get("carryover_ratio"),
            row.get("reward_beta"),
            row.get("seed"),
            row.get("instance_id"),
            row.get("method"),
        )
        deduped[key] = row
    duplicate_rows = len(filtered) - len(deduped)
    filtered = list(deduped.values())
    after_size = count_by(filtered, "size")

    print(f"Rows by size before filtering: {before_size}")
    print(f"Rows by size after filtering: {after_size}")
    print(f"Dropped unknown/invalid size rows: {len(unknown_rows)}")
    print(f"Dropped FIFO/excluded method rows: {len(fifo_rows)}")
    print(f"Dropped duplicate case-method rows: {duplicate_rows}")
    print(f"Final paper methods: {sorted({row.get('method') for row in filtered})}")
    print(f"Final paper sizes: {sorted({row.get('size') for row in filtered})}")

    audit = list(source_rows)
    audit.extend(
        [
            {"audit_type": "rows_by_size_before", "name": key, "value": value, "notes": "raw selected detail rows"}
            for key, value in before_size.items()
        ]
    )
    audit.extend(
        [
            {"audit_type": "rows_by_size_after", "name": key, "value": value, "notes": "valid no-FIFO paper rows"}
            for key, value in after_size.items()
        ]
    )
    audit.extend(
        [
            {"audit_type": "filter", "name": "unknown_or_invalid_size", "value": len(unknown_rows), "notes": "excluded from every statistic"},
            {"audit_type": "filter", "name": "excluded_methods", "value": len(fifo_rows), "notes": "methods=" + " ".join(args.exclude_methods)},
            {"audit_type": "filter", "name": "duplicate_case_method", "value": duplicate_rows, "notes": "last selected row retained"},
            {"audit_type": "final", "name": "rows", "value": len(filtered), "notes": args.output_tag},
            {"audit_type": "final", "name": "methods", "value": len({row.get('method') for row in filtered}), "notes": " ".join(sorted({row.get('method') for row in filtered}))},
            {"audit_type": "final", "name": "sizes", "value": len({row.get('size') for row in filtered}), "notes": " ".join(sorted({row.get('size') for row in filtered}))},
        ]
    )
    return filtered, audit


def group_cases(rows: Sequence[dict], methods: Sequence[str]) -> Dict[str, Dict[str, dict]]:
    allowed = set(methods)
    cases: Dict[str, Dict[str, dict]] = {}
    for row in rows:
        if row.get("method") in allowed:
            cases.setdefault(case_id(row), {})[row["method"]] = row
    return cases


def build_case_rows(rows: Sequence[dict], args) -> List[dict]:
    output_methods = list(dict.fromkeys([*args.external_methods, *args.include_weak_baselines]))
    reference_methods = list(args.external_methods)
    if args.weak_baselines_in_reference:
        reference_methods.extend(args.include_weak_baselines)
    cases = group_cases(rows, output_methods)
    labels = {cid: f"C{idx + 1}" for idx, cid in enumerate(sorted(cases))}
    out = []
    for cid, methods in sorted(cases.items()):
        reference = {name: methods[name] for name in reference_methods if name in methods}
        if not reference:
            continue
        best = min(fnum(row["Cmax"]) for row in reference.values())
        ranked = sorted(reference.items(), key=lambda item: (fnum(item[1]["Cmax"]), item[0]))
        ranks = {method: idx + 1 for idx, (method, _) in enumerate(ranked)}
        hgcr = methods.get("HGCR-PPO")
        for method, row in methods.items():
            cmax = fnum(row["Cmax"])
            record = {
                "case_label": labels[cid],
                "case_id": cid,
                "size": row["size"],
                "arrival_intensity": row.get("arrival_intensity", ""),
                "carryover_ratio": row.get("carryover_ratio", ""),
                "reward_beta": row.get("reward_beta", ""),
                "seed": row.get("seed", ""),
                "instance_id": row.get("instance_id", ""),
                "method": method,
                "Cmax": cmax,
                "Cmax_best_non_fifo": best,
                "ARPD_no_fifo": (cmax - best) / max(best, 1e-8) * 100.0,
                "rank_no_fifo": ranks.get(method, ""),
            }
            for baseline in ["MLP-Ranker", "GreedyECT", "Lookahead", "MinLoad"]:
                base = methods.get(baseline)
                record[f"improvement_vs_{baseline.replace('-', '_')}"] = (
                    (fnum(base["Cmax"]) - fnum(hgcr["Cmax"])) / max(fnum(base["Cmax"]), 1e-8) * 100.0
                    if base and hgcr
                    else ""
                )
            out.append(record)
    return out


def method_summary(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in rows:
        key = (row["size"], row.get("arrival_intensity", ""), row.get("carryover_ratio", ""), row.get("reward_beta", ""), row["method"])
        groups.setdefault(key, []).append(row)
    out = []
    for (size, arrival, carryover, beta, method), values in sorted(groups.items()):
        cmax = [fnum(row["Cmax"]) for row in values]
        out.append(
            {
                "size": size,
                "arrival_intensity": arrival,
                "carryover_ratio": carryover,
                "reward_beta": beta,
                "method": method,
                "Cmax_mean": mean(cmax),
                "Cmax_std": pstdev(cmax) if len(cmax) > 1 else 0.0,
                "average_completion_time_mean": mean(fnum(row.get("average_completion_time")) for row in values),
                "average_waiting_time_mean": mean(fnum(row.get("average_waiting_time")) for row in values),
                "machine_utilization_mean": mean(fnum(row.get("machine_utilization")) for row in values),
                "load_balance_std_mean": mean(fnum(row.get("load_balance_std")) for row in values),
                "runtime_seconds_mean": mean(fnum(row.get("runtime_seconds")) for row in values),
                "valid_schedule_rate": mean(1.0 if str(row.get("valid_schedule")).lower() == "true" else 0.0 for row in values),
                "n_instances": len(values),
            }
        )
    return out


def arpd_summary(case_rows: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in case_rows:
        if row.get("rank_no_fifo") == "":
            continue
        key = (row["size"], row["method"], row["arrival_intensity"], row["carryover_ratio"], row["reward_beta"])
        groups.setdefault(key, []).append(row)
    out = []
    for (size, method, arrival, carryover, beta), values in sorted(groups.items()):
        arpd = [fnum(row["ARPD_no_fifo"]) for row in values]
        ranks = [fnum(row["rank_no_fifo"]) for row in values]
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
                "n_instances": len(values),
            }
        )
    return out


def scale_summary(case_rows: Sequence[dict], valid_sizes: Sequence[str]) -> List[dict]:
    valid = set(valid_sizes)
    groups: Dict[tuple, List[dict]] = {}
    for row in case_rows:
        if row["size"] not in valid or row.get("rank_no_fifo") == "":
            continue
        groups.setdefault((row["size"], row["method"]), []).append(row)
        groups.setdefault(("all", row["method"]), []).append(row)
    out = []
    for (size, method), values in sorted(groups.items()):
        cmax = [fnum(row["Cmax"]) for row in values]
        arpd = [fnum(row["ARPD_no_fifo"]) for row in values]
        ranks = [fnum(row["rank_no_fifo"]) for row in values]
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
                "n_instances": len(values),
            }
        )
    return out


def paired_p_value(diffs: List[float]) -> tuple[str, float | str]:
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


def significance(rows: Sequence[dict], methods: Sequence[str]) -> List[dict]:
    cases = group_cases(rows, methods)
    out = []
    for baseline in SIG_BASELINES:
        diffs = []
        for values in cases.values():
            if "HGCR-PPO" in values and baseline in values:
                diffs.append(fnum(values["HGCR-PPO"]["Cmax"]) - fnum(values[baseline]["Cmax"]))
        test, p_value = paired_p_value(diffs)
        avg = mean(diffs) if diffs else 0.0
        out.append(
            {
                "comparison": f"HGCR-PPO_vs_{baseline}",
                "baseline_method": baseline,
                "test_name": test,
                "n_pairs": len(diffs),
                "mean_diff": avg,
                "median_diff": median(diffs) if diffs else 0.0,
                "p_value": p_value,
                "significant": p_value != "" and float(p_value) < 0.05,
                "effect_direction": "HGCR_better" if avg < 0 else "baseline_better_or_tie",
            }
        )
    return out


def case_mapping(case_rows: Sequence[dict]) -> List[dict]:
    mapping = {}
    for row in case_rows:
        mapping.setdefault(
            row["case_label"],
            {key: row.get(key, "") for key in ["case_label", "case_id", "size", "arrival_intensity", "carryover_ratio", "seed", "instance_id"]},
        )
    return [mapping[key] for key in sorted(mapping, key=lambda value: int(value.lstrip("C")))]


def run(args):
    selected = select_detail_files(args)
    print("Selected detail files:")
    all_rows = []
    source_audit = []
    for path in selected:
        rows = read_csv(path)
        print(f"  - {path}: {len(rows)} rows")
        all_rows.extend(rows)
        source_audit.append({"audit_type": "source_file", "name": str(path), "value": len(rows), "notes": "selected detail input"})
    if not selected:
        print("Warning: no size-aware Stage G detail file found.")

    filtered, audit = audit_and_filter(all_rows, args, source_audit)
    suffix = token()
    paths = output_paths(Path(args.output_dir), suffix)
    for path in paths.values():
        print(f"Planned output: {path}")
    if args.dry_run:
        print("Dry run enabled: no files will be written.")
        return paths

    cases = build_case_rows(filtered, args)
    summary = method_summary(filtered)
    arpd = arpd_summary(cases)
    scale = scale_summary(cases, args.valid_sizes)
    sig = significance(filtered, args.external_methods)
    mapping = case_mapping(cases)
    if args.no_write:
        print(f"No-write enabled: {len(summary)} summary, {len(arpd)} ARPD, {len(scale)} scale rows prepared.")
        return paths

    write_csv(paths["summary"], summary, ["size", "arrival_intensity", "carryover_ratio", "reward_beta", "method", "Cmax_mean", "Cmax_std", "average_completion_time_mean", "average_waiting_time_mean", "machine_utilization_mean", "load_balance_std_mean", "runtime_seconds_mean", "valid_schedule_rate", "n_instances"])
    write_csv(paths["arpd"], arpd, ["size", "method", "arrival_intensity", "carryover_ratio", "reward_beta", "ARPD_mean", "ARPD_std", "rank_mean", "rank_std", "n_instances"])
    write_csv(paths["significance"], sig, ["comparison", "baseline_method", "test_name", "n_pairs", "mean_diff", "median_diff", "p_value", "significant", "effect_direction"])
    write_csv(paths["case"], cases, ["case_label", "case_id", "size", "arrival_intensity", "carryover_ratio", "reward_beta", "seed", "instance_id", "method", "Cmax", "Cmax_best_non_fifo", "ARPD_no_fifo", "rank_no_fifo", "improvement_vs_MLP_Ranker", "improvement_vs_GreedyECT", "improvement_vs_Lookahead", "improvement_vs_MinLoad"])
    write_csv(paths["scale"], scale, ["size", "method", "Cmax_mean", "Cmax_std", "ARPD_mean", "ARPD_std", "rank_mean", "rank_std", "n_instances"])
    write_csv(paths["audit"], audit, ["audit_type", "name", "value", "notes"])
    write_csv(paths["mapping"], mapping, ["case_label", "case_id", "size", "arrival_intensity", "carryover_ratio", "seed", "instance_id"])
    print(f"Saved clean no-FIFO v3 paper results to {args.output_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(PAPER_DIR))
    parser.add_argument("--output_dir", default=str(PAPER_DIR))
    parser.add_argument("--detail_file")
    parser.add_argument("--detail_files", nargs="+")
    parser.add_argument("--use_latest_detail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude_unknown_size", action="store_true", default=True)
    parser.add_argument("--valid_sizes", nargs="+", default=VALID_SIZES)
    parser.add_argument("--exclude_methods", nargs="+", default=["FIFO"])
    parser.add_argument("--external_methods", nargs="+", default=EXTERNAL_METHODS)
    parser.add_argument("--include_weak_baselines", nargs="*", default=WEAK_BASELINES)
    parser.add_argument("--weak_baselines_in_reference", action="store_true")
    parser.add_argument("--output_tag", default="no_fifo_clean")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
