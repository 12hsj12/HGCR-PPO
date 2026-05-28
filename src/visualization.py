"""Gantt-chart visualization for split scheduling results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def plot_gantt(env, save_path: str = "data/results/gantt.png") -> str:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    machines = sorted(env.instance.machines, key=lambda m: (m.process_type, m.machine_id))
    machine_to_y: Dict[str, int] = {machine.machine_id: idx for idx, machine in enumerate(machines)}
    jobs = sorted(env.scheduled_jobs)
    cmap = plt.get_cmap("tab20")
    job_to_color = {job_id: cmap(idx % 20) for idx, job_id in enumerate(jobs)}

    height = max(4.0, 0.42 * len(machines) + 1.8)
    width = 12.0
    fig, ax = plt.subplots(figsize=(width, height))

    for subtask in env.subtasks:
        y = machine_to_y[subtask.machine_id]
        ax.barh(
            y=y,
            width=subtask.duration,
            left=subtask.start_time,
            height=0.72,
            color=job_to_color[subtask.job_id],
            edgecolor="black",
            linewidth=0.5,
        )
        label = f"{subtask.job_id}\n{subtask.ratio:.2f}"
        ax.text(
            subtask.start_time + subtask.duration / 2.0,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            clip_on=True,
        )

    ax.set_yticks(list(machine_to_y.values()))
    ax.set_yticklabels([machine.machine_id for machine in machines])
    ax.set_xlabel("Time")
    ax.set_ylabel("Production line")
    ax.set_title(f"Gantt chart - {env.instance.name}")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)

