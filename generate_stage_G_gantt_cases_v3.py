"""Generate the final no-FIFO Stage G representative Gantt comparison."""

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
SUPPORTED_BASELINES = ["MLP-Ranker", "GreedyECT", "Lookahead", "MinLoad"]


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}_no_fifo_v3"


def latest(root: Path, pattern: str) -> Path | None:
    paths = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    return paths[-1] if paths else None


def read_csv(path: Path | None) -> List[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def method_stem(method: str) -> str:
    return method.replace("-", "_").replace(" ", "_")


def validate_baselines(primary: str, candidates: List[str], excluded: set[str]) -> List[str]:
    ordered = list(dict.fromkeys([primary, *candidates]))
    invalid = [method for method in ordered if method not in SUPPORTED_BASELINES]
    if invalid:
        raise ValueError(f"Unsupported candidate baselines: {invalid}. Allowed: {SUPPORTED_BASELINES}")
    return [method for method in ordered if method not in excluded]


def select_case(detail: List[dict], trace: List[dict], references: List[str], excluded: set[str]) -> dict | None:
    trace_keys = {
        (row.get("scenario_run_id"), row.get("instance_id"), row.get("method"))
        for row in trace
        if row.get("method") not in excluded
    }
    grouped: Dict[tuple, Dict[str, dict]] = {}
    for row in detail:
        if row.get("method") in excluded or str(row.get("valid_schedule")).lower() != "true":
            continue
        grouped.setdefault((row.get("scenario_run_id"), row.get("instance_id")), {})[row.get("method")] = row
    for reference in references:
        candidates = []
        for key, methods in grouped.items():
            if reference not in methods or "HGCR-PPO" not in methods:
                continue
            if (key[0], key[1], reference) not in trace_keys or (key[0], key[1], "HGCR-PPO") not in trace_keys:
                continue
            gap = fnum(methods[reference].get("Cmax")) - fnum(methods["HGCR-PPO"].get("Cmax"))
            if gap > 0:
                candidates.append({"key": key, "methods": methods, "reference_method": reference, "gap": gap})
        if candidates:
            print(f"Selected candidate baseline: {reference}")
            return max(candidates, key=lambda item: item["gap"])
        print(f"Warning: no positive-gap valid case found for baseline={reference}; trying next candidate.")
    return None


def trace_rows(trace: List[dict], case: dict, method: str) -> List[dict]:
    run_id, instance_id = case["key"]
    return [
        row
        for row in trace
        if row.get("scenario_run_id") == run_id and row.get("instance_id") == instance_id and row.get("method") == method
    ]


def save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def draw_method(ax, rows: List[dict], stats: dict, method: str) -> None:
    import matplotlib.pyplot as plt

    machines = sorted({row["machine_id"] for row in rows})
    machine_y = {machine: idx for idx, machine in enumerate(machines)}
    jobs = sorted({row["job_id"] for row in rows})
    cmap = plt.get_cmap("tab20")
    colors = {job: cmap(idx % 20) for idx, job in enumerate(jobs)}
    for row in rows:
        ax.barh(
            machine_y[row["machine_id"]],
            fnum(row.get("duration")),
            left=fnum(row.get("start_time")),
            height=0.72,
            color=colors[row["job_id"]],
            edgecolor="black",
            linewidth=0.3,
        )
    cmax = fnum(stats.get("Cmax"))
    ax.axvline(cmax, color="#C83E36", linestyle="--", linewidth=1.2)
    if machines:
        ax.text(cmax, len(machines) - 0.35, f"Cmax={cmax:.1f}", color="#C83E36", ha="right", va="top", fontsize=8)
    if rows:
        tail = max(rows, key=lambda row: fnum(row.get("end_time")))
        label = "reduced tail" if method == "HGCR-PPO" else "bottleneck tail"
        ax.annotate(
            label,
            xy=(fnum(tail.get("end_time")), machine_y[tail["machine_id"]]),
            xytext=(-62, 14),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "linewidth": 0.7},
            fontsize=7,
        )
    ax.set_yticks(list(machine_y.values()))
    ax.set_yticklabels([MACHINE_NAMES.get(machine, machine) for machine in machines])
    ax.set_title(
        f"{method}: Cmax={cmax:.1f}, util={fnum(stats.get('machine_utilization')):.2f}, wait={fnum(stats.get('average_waiting_time')):.1f}",
        loc="left",
        fontsize=9,
    )
    ax.set_xlabel("Time")
    ax.grid(axis="x", alpha=0.25)


def render(case: dict, trace: List[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    reference = case["reference_method"]
    reference_stem = method_stem(reference)
    stems = {"HGCR-PPO": "gantt_case1_HGCR_PPO", reference: f"gantt_case1_{reference_stem}"}
    for method in [reference, "HGCR-PPO"]:
        fig, ax = plt.subplots(figsize=(10.5, 3.1))
        draw_method(ax, trace_rows(trace, case, method), case["methods"][method], method)
        fig.tight_layout()
        save(fig, out_dir, stems[method])
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True)
    for ax, method in zip(axes, [reference, "HGCR-PPO"]):
        draw_method(ax, trace_rows(trace, case, method), case["methods"][method], method)
    ref_row = case["methods"][reference]
    fig.suptitle("Representative scheduling case", y=1.01)
    fig.tight_layout()
    save(fig, out_dir, f"gantt_case1_comparison_{reference_stem}_vs_HGCR")
    plt.close(fig)

    metadata = {
        "scenario_run_id": case["key"][0],
        "instance_id": case["key"][1],
        "reference_method": reference,
        "reference_Cmax": fnum(case["methods"][reference].get("Cmax")),
        "HGCR_PPO_Cmax": fnum(case["methods"]["HGCR-PPO"].get("Cmax")),
        "gap_vs_reference": case["gap"],
        "valid_schedule": True,
        "why_selected": f"largest positive {reference} minus HGCR-PPO Cmax gap with complete valid traces",
    }
    (out_dir / "gantt_case1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "gantt_case1_caption.txt").write_text(
        "\n".join(
            [
                f"Scenario run: {metadata['scenario_run_id']}",
                f"Instance: {metadata['instance_id']}",
                f"{reference} Cmax: {metadata['reference_Cmax']:.3f}",
                f"HGCR-PPO Cmax: {metadata['HGCR_PPO_Cmax']:.3f}",
                f"Cmax reduction: {metadata['gap_vs_reference']:.3f}",
                f"Selected as the largest valid {reference} versus HGCR-PPO gap with complete schedule traces.",
            ]
        ),
        encoding="utf-8",
    )


def run(args):
    excluded = set(args.exclude_methods)
    references = validate_baselines(args.primary_baseline, args.candidate_baselines, excluded)
    if not references:
        raise ValueError("All candidate baselines were excluded; at least one non-FIFO baseline is required.")
    root = Path(args.paper_dir)
    detail_path = Path(args.detail_file) if args.detail_file else latest(root, "stage_G_method_comparison_detail__*.csv")
    trace_path = Path(args.trace_file) if args.trace_file else latest(root, "schedule_trace__*.csv")
    out_dir = Path(args.output_dir) / token()
    print(f"Detail input: {detail_path if detail_path else 'missing'}")
    print(f"Trace input: {trace_path if trace_path else 'missing'}")
    print(f"Primary baseline: {args.primary_baseline}")
    print(f"Candidate baseline order: {references}")
    print(f"Excluded methods: {sorted(excluded)}")
    print(f"Planned output dir: {out_dir}")
    if args.dry_run:
        print("Dry run enabled: no Gantt files will be written.")
        return out_dir
    detail = [row for row in read_csv(detail_path) if row.get("method") not in excluded]
    trace = [row for row in read_csv(trace_path) if row.get("method") not in excluded]
    case = select_case(detail, trace, references, excluded)
    if case is None:
        print(f"Warning: no valid traced positive-gap case found for candidate baselines {references}.")
        return out_dir
    if args.no_write:
        print(f"No-write enabled: selected {case['key']} gap={case['gap']:.3f}.")
        return out_dir
    render(case, trace, out_dir)
    print(f"Saved v3 no-FIFO Gantt case to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_dir", default=str(PAPER_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--detail_file")
    parser.add_argument("--trace_file")
    parser.add_argument("--primary_baseline", default="MLP-Ranker", choices=SUPPORTED_BASELINES)
    parser.add_argument("--candidate_baselines", nargs="+", default=["MLP-Ranker"])
    parser.add_argument("--exclude_methods", nargs="+", default=["FIFO"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
