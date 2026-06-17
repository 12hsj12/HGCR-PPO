# HGCR-PPO

Heuristic-guided Graph Candidate Ranking Proximal Policy Optimization

HGCR-PPO is a research-code project for task splitting, non-identical parallel production lines, and multi-period rolling scheduling in steel processing centers. The main optimization target is the actual rolling makespan, `Cmax_roll`.

The repository has evolved from the original TSG-PPO baseline into a staged experimental pipeline. The current paper-facing branch focuses on Stage G / Stage G-Plus: dynamic rolling scenarios, HGCR-PPO rule-selection training, no-FIFO external paper statistics, and refined v3 paper figures.

## Current Status

The current codebase includes:

- Instance generation and fixed train/val/test datasets.
- Rolling scheduling environment with task splitting.
- Heuristic baselines and metrics.
- Candidate generation and MLP/GNN ranker utilities.
- Dynamic Stage G HGCR-PPO training.
- No-FIFO v3 paper result summarization.
- Refined v3 paper figures and representative Gantt cases.

The PPO action space in Stage G remains fixed at four internal dispatching rules:

0. Arrival-order rule, implemented by the original FIFO order.
1. GreedyECT.
2. Lookahead.
3. MLP-Ranker soft_ce.

Important terminology: external paper comparisons no longer use FIFO as a baseline. The internal arrival-order action is retained for HGCR-PPO, but figures and action tables display it as `Arrival-order rule`.

## Repository Layout

```text
HGCR-PPO/
  run_baselines.py
  run_hgcr_dynamic_ppo.py
  evaluate_stage_G_dynamic_baselines.py
  summarize_stage_G_paper_results_v3.py
  plot_stage_G_paper_figures_v3.py
  generate_stage_G_gantt_cases_v3.py
  audit_stage_G_runs_v3.py
  instance_manager.py
  dynamic_rolling_scenarios.py
  candidate_generator.py
  mlp_models.py
  gnn_ranker_models.py
  src/
    baselines/
    envs/
    evaluation/
    instances/
  ppo/
  docs/
  tests/
```

Main output directories:

```text
data/results/stage_G/hgcr_dynamic_ppo/runs/
data/results/stage_G/summary/
data/results/stage_G/paper_results/
data/results/stage_G/paper_figures/
data/results/stage_G/gantt_cases/
```

Stage G scripts should not write to Stage E or Stage F output directories.

## Installation

On Windows:

```powershell
git clone https://github.com/12hsj12/HGCR-PPO.git
cd HGCR-PPO
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA training, install the PyTorch build that matches the remote machine's CUDA runtime if needed.

## Basic Baselines

The legacy baseline entry remains compatible:

```powershell
python run_baselines.py
```

It runs heuristic dispatching methods and writes result artifacts under `data/results/`.

## Stage A Fixed Evaluation

Stage A builds a stable, reproducible evaluation foundation:

```powershell
python run_stage_A.py --sizes small medium large
python evaluate_methods.py --size small --split test
python check_split_effect.py --size small --split test
```

Fixed instances are stored under:

```text
data/instances/fixed/train/
data/instances/fixed/val/
data/instances/fixed/test/
```

## Stage G HGCR-PPO Training

The Stage G dynamic PPO entry is:

```powershell
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity medium --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo --disable_early_stop --eval_interval 50
```

For paper runs, use:

- `--disable_early_stop` to ensure the intended training horizon is reached.
- `--eval_interval 50` for dense convergence curves.
- `--reward_mode util_plus_cmax`.
- `--reward_beta 5.0` for the main configuration.
- `--baseline_method fifo` as the internal reward baseline only.

The training script writes per-run artifacts to:

```text
data/results/stage_G/hgcr_dynamic_ppo/runs/{run_id}/
  eval_history*.csv
  action_history*.csv
  action_stage_summary*.csv
  eval_summary.csv
  train_log.csv
  manifest.json
  hgcr_dynamic_ppo.pt
```

## Stage G Paper v3 Pipeline

The v3 pipeline is the current paper-facing result path.

1. Audit missing or invalid training runs:

```powershell
python audit_stage_G_runs_v3.py --profile paper_v3 --write_missing_commands --exclude_existing_valid --output_bat data/results/stage_G/summary/missing_runs_paper_v3.bat --output_csv data/results/stage_G/summary/missing_runs_paper_v3.csv
```

2. Run only the generated missing commands on the high-performance machine:

```powershell
cmd /c data\results\stage_G\summary\missing_runs_paper_v3.bat
```

3. Evaluate non-FIFO external baselines:

```powershell
python evaluate_stage_G_dynamic_baselines.py --methods Random SPT LPT GreedyECT Lookahead MinLoad MLP-Ranker HGCR-PPO --save_schedule_trace
```

4. Build clean no-FIFO v3 paper statistics:

```powershell
python summarize_stage_G_paper_results_v3.py --use_latest_detail --exclude_unknown_size --exclude_methods FIFO --external_methods HGCR-PPO MLP-Ranker GreedyECT Lookahead MinLoad --include_weak_baselines Random SPT LPT --output_tag no_fifo_clean
```

The summarizer:

- Uses the latest size-aware detail file by default.
- Supports `--detail_file` and `--detail_files` for exact input control.
- Drops rows whose `size` is not `small`, `medium`, or `large`.
- Removes external FIFO rows.
- Recomputes ARPD, rank, and significance with no-FIFO references.
- Builds `all` only from `small`, `medium`, and `large`.

5. Generate refined no-FIFO v3 figures:

```powershell
python plot_stage_G_paper_figures_v3.py
```

Default refined figures include:

- `Fig_2_training_convergence_small_no_fifo`
- `Fig_2b_training_reward_convergence_small_no_fifo`
- `Fig_3_beta_sensitivity_training_curves_no_fifo`
- `Fig_4_action_ratio_evolution_no_fifo_label`
- `Fig_5_case_performance_curves_no_fifo`
- `Fig_6a_distribution_all_methods_no_fifo`
- `Fig_8_scale_generalization_no_fifo`
- `Fig_A1_dynamic_heatmap_3x3_no_fifo`
- `Fig_A2_beta_final_summary_no_fifo`
- `Fig_A3_win_tie_loss_no_fifo`
- `Fig_A4_arpd_rank_no_fifo`

`Fig_A4_significance_table_no_fifo.csv` and `Fig_A4_arpd_rank_source_no_fifo.csv` are exported alongside the figures.

6. Generate representative no-FIFO Gantt cases:

```powershell
python generate_stage_G_gantt_cases_v3.py
```

The default main-text Gantt comparison is MLP-Ranker vs HGCR-PPO. Optional candidates can be supplied:

```powershell
python generate_stage_G_gantt_cases_v3.py --primary_baseline MLP-Ranker --candidate_baselines MLP-Ranker GreedyECT --exclude_methods FIFO
```

## No-FIFO Paper Policy

For paper-facing external comparisons:

- Do not include FIFO in method comparison figures.
- Do not include FIFO in ARPD, rank, or significance statistics.
- Do not use unknown-size rows.
- Use `Arrival-order rule` only for HGCR-PPO internal action visualizations.

Current external main methods:

- HGCR-PPO
- MLP-Ranker
- GreedyECT
- Lookahead
- MinLoad

Weak baselines are optional diagnostic rows:

- Random
- SPT
- LPT

## Tests and Lightweight Checks

Run unit tests:

```powershell
python -m pytest -q
```

Lightweight script checks:

```powershell
python -m py_compile run_hgcr_dynamic_ppo.py
python -m py_compile evaluate_stage_G_dynamic_baselines.py
python -m py_compile summarize_stage_G_paper_results_v3.py
python -m py_compile plot_stage_G_paper_figures_v3.py
python -m py_compile generate_stage_G_gantt_cases_v3.py
python -m py_compile audit_stage_G_runs_v3.py
```

Dry-run checks:

```powershell
python audit_stage_G_runs_v3.py --profile paper_v3 --write_missing_commands --dry_run
python summarize_stage_G_paper_results_v3.py --use_latest_detail --exclude_unknown_size --exclude_methods FIFO --dry_run --no_write
python plot_stage_G_paper_figures_v3.py --dry_run --no_write
python generate_stage_G_gantt_cases_v3.py --dry_run --no_write
```

## Notes

This repository is research code. Result CSVs, checkpoints, and generated figures can be large and are normally treated as runtime artifacts rather than source files.
