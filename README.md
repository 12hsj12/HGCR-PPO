# TSG-PPO

TSG-PPO is a research-code project for task splitting, unrelated/non-identical parallel production lines, and multi-period rolling scheduling.

The current version focuses on a reliable scheduling foundation before adding deep reinforcement learning. It includes instance generation, an RL-compatible rolling scheduling environment, heuristic baselines, evaluation metrics, and Gantt-chart visualization.

## Current Features

- Reproducible small, medium, and large instance generation.
- Rolling scheduling environment with action `(job_id, split_num)`.
- Machine selection by estimated earliest completion time.
- Split-ratio calculation by inverse processing time.
- Heuristic baselines:
  - Random
  - FIFO
  - SPT
  - LPT
  - MinCandidateLoad
  - GreedyECT
- Evaluation metrics:
  - rolling makespan
  - average completion time
  - average waiting time
  - machine utilization
  - load balance standard deviation
  - split task ratio
  - total split count
- Gantt-chart output for schedule validation.

## Project Structure

```text
TSG-PPO/
├── run_baselines.py
├── requirements.txt
├── README.md
├── scripts/
│   └── run_baselines.py
├── src/
│   ├── core.py
│   ├── instances/
│   │   └── instance_generator.py
│   ├── envs/
│   │   └── rolling_scheduling_env.py
│   ├── baselines/
│   │   └── heuristics.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── visualization.py
└── tests/
```

## Installation

On Windows, clone the repository and create a virtual environment:

```bash
git clone https://github.com/12hsj12/TSG-PPO.git
cd TSG-PPO
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Baselines

```bash
python run_baselines.py
```

The script generates small, medium, and large instances, runs all heuristic baselines, prints a metrics table, and saves Gantt charts for the small instance.

## Outputs

Generated visualization files are saved under:

```text
data/results/
```

These files are runtime outputs and are ignored by Git by default.

## Tests

```bash
python -m pytest -q
```

The tests cover reproducible instance generation, key feasibility conditions, and metric ranges.
