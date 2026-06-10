"""Stage G-Plus v2 representative Gantt figures from schedule_trace."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PAPER_DIR = Path("data/results/stage_G/paper_results")
OUTPUT_DIR = Path("data/results/stage_G/gantt_cases")
MACHINE_NAMES = {
    "sl_m1": "Slitting-1",
    "sl_m2": "Slitting-2",
    "sl_m3": "Slitting-3",
    "cu_m1": "Cutting-1",
    "cu_m2": "Cutting-2",
    "co_m1": "Composite-1",
    "co_m2": "Composite-2",
}


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    return matches[-1] if matches else None


def read_csv(path: Path | None) -> List[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def select_case(detail: List[dict], trace: List[dict]) -> dict | None:
    trace_keys = {(r["scenario_run_id"], r["instance_id"], r["method"]) for r in trace}
    grouped: Dict[tuple, Dict[str, dict]] = {}
    for row in detail:
        if str(row.get("valid_schedule")).lower() != "true":
            continue
        grouped.setdefault((row["scenario_run_id"], row["instance_id"]), {})[row["method"]] = row
    fifo_candidates = []
    mlp_candidates = []
    for key, methods in grouped.items():
        if "HGCR-PPO" not in methods:
            continue
        if (key[0], key[1], "HGCR-PPO") not in trace_keys:
            continue
        if "FIFO" in methods and (key[0], key[1], "FIFO") in trace_keys:
            gap = fnum(methods["FIFO"]["Cmax"]) - fnum(methods["HGCR-PPO"]["Cmax"])
            if gap > 0:
                fifo_candidates.append({"key": key, "methods": methods, "gap": gap, "reference_method": "FIFO"})
        if "MLP-Ranker" in methods and (key[0], key[1], "MLP-Ranker") in trace_keys:
            gap = fnum(methods["MLP-Ranker"]["Cmax"]) - fnum(methods["HGCR-PPO"]["Cmax"])
            if gap > 0:
                mlp_candidates.append({"key": key, "methods": methods, "gap": gap, "reference_method": "MLP-Ranker"})
    if fifo_candidates:
        return max(fifo_candidates, key=lambda c: c["gap"])
    return max(mlp_candidates, key=lambda c: c["gap"]) if mlp_candidates else None


def rows_for(trace: List[dict], case: dict, method: str) -> List[dict]:
    run_id, instance_id = case["key"]
    return [r for r in trace if r["scenario_run_id"] == run_id and r["instance_id"] == instance_id and r["method"] == method]


def save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def draw_method(ax, rows: List[dict], stats: dict, method: str) -> None:
    import matplotlib.pyplot as plt

    machines = sorted({r["machine_id"] for r in rows})
    y = {m: i for i, m in enumerate(machines)}
    jobs = sorted({r["job_id"] for r in rows})
    cmap = plt.get_cmap("tab20")
    colors = {j: cmap(i % 20) for i, j in enumerate(jobs)}
    for r in rows:
        ax.barh(y[r["machine_id"]], fnum(r["duration"]), left=fnum(r["start_time"]), height=0.72, color=colors[r["job_id"]], edgecolor="black", linewidth=0.3)
    cmax = fnum(stats["Cmax"])
    ax.axvline(cmax, color="#D14B3F", linestyle="--", linewidth=1.1)
    ax.text(cmax, len(machines) - 0.35, f"Cmax={cmax:.1f}", color="#D14B3F", ha="right", va="top", fontsize=8)
    if rows:
        tail = max(rows, key=lambda r: fnum(r["end_time"]))
        ax.annotate(
            "bottleneck tail",
            xy=(fnum(tail["end_time"]), y[tail["machine_id"]]),
            xytext=(-62, 14),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "linewidth": 0.7},
            fontsize=7,
        )
    ax.set_yticks(list(y.values()))
    ax.set_yticklabels([MACHINE_NAMES.get(m, m) for m in machines])
    ax.set_title(f"{method}: Cmax={cmax:.1f}, util={fnum(stats.get('machine_utilization')):.2f}, wait={fnum(stats.get('average_waiting_time')):.1f}", loc="left", fontsize=9)
    ax.set_xlabel("Time")
    ax.grid(axis="x", alpha=0.25)


def render(case: dict, trace: List[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    reference = case.get("reference_method", "FIFO")
    methods = [reference, "HGCR-PPO"] + (["MLP-Ranker"] if reference != "MLP-Ranker" and "MLP-Ranker" in case["methods"] and rows_for(trace, case, "MLP-Ranker") else [])
    for method in methods:
        fig, ax = plt.subplots(figsize=(10.5, 3.1))
        draw_method(ax, rows_for(trace, case, method), case["methods"][method], method)
        fig.tight_layout()
        save(fig, out_dir, f"gantt_case1_{method.replace('-', '_')}")
        plt.close(fig)
    for stem, selected in [("gantt_case1_comparison_maintext", [reference, "HGCR-PPO"]), ("gantt_case1_comparison", methods)]:
        fig, axes = plt.subplots(len(selected), 1, figsize=(10.5, 2.8 * len(selected)), sharex=True)
        if len(selected) == 1:
            axes = [axes]
        for ax, method in zip(axes, selected):
            draw_method(ax, rows_for(trace, case, method), case["methods"][method], method)
        ref_row = case["methods"][reference]
        fig.suptitle(f"Representative case: {ref_row['arrival_intensity']} arrival, {ref_row['carryover_ratio']} carryover", y=1.02)
        fig.tight_layout()
        save(fig, out_dir, stem)
        plt.close(fig)
    metadata = {
        "selected_scenario": case["key"][0],
        "instance_id": case["key"][1],
        "reference_method": reference,
        "reference_Cmax": fnum(case["methods"][reference]["Cmax"]),
        "FIFO_Cmax": fnum(case["methods"].get("FIFO", {}).get("Cmax")) if "FIFO" in case["methods"] else None,
        "HGCR_PPO_Cmax": fnum(case["methods"]["HGCR-PPO"]["Cmax"]),
        "MLP_Ranker_Cmax": fnum(case["methods"].get("MLP-Ranker", {}).get("Cmax")) if "MLP-Ranker" in case["methods"] else None,
        "gap_vs_reference": case["gap"],
        "why_selected": f"largest valid Cmax gap {reference} - HGCR-PPO with complete schedule_trace",
    }
    (out_dir / "gantt_case1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "gantt_case1_caption.txt").write_text(
        "\n".join(
            [
                f"Selected scenario: {metadata['selected_scenario']}",
                f"Instance id: {metadata['instance_id']}",
                f"Reference method: {metadata['reference_method']}",
                f"Reference Cmax: {metadata['reference_Cmax']:.3f}",
                f"HGCR-PPO Cmax: {metadata['HGCR_PPO_Cmax']:.3f}",
                f"gap_vs_reference: {metadata['gap_vs_reference']:.3f}",
                f"MLP-Ranker Cmax: {metadata['MLP_Ranker_Cmax']}",
                "Why selected: largest valid FIFO minus HGCR-PPO Cmax gap with complete trace.",
            ]
        ),
        encoding="utf-8",
    )


def run(args):
    root = Path(args.paper_dir)
    detail_path = latest(root, "stage_G_method_comparison_detail__*.csv")
    trace_path = latest(root, "schedule_trace__*.csv")
    out_dir = Path(args.output_dir) / token()
    print(f"Detail input: {detail_path if detail_path else 'missing'}")
    print(f"Trace input: {trace_path if trace_path else 'missing'}")
    print(f"Planned output dir: {out_dir}")
    if args.dry_run:
        print("Dry run enabled: no Gantt files will be written.")
        return out_dir
    detail = read_csv(detail_path)
    trace = read_csv(trace_path)
    case = select_case(detail, trace)
    if case is None:
        print("Warning: no valid traced case found for FIFO vs HGCR-PPO.")
        return out_dir
    if args.no_write:
        print(f"No-write enabled: selected {case['key']} gap={case['gap']:.3f}, no files written.")
        return out_dir
    render(case, trace, out_dir)
    print(f"Saved v2 Gantt case to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_dir", default=str(PAPER_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
