# 阶段 G Plus 论文级结果包命令

本文件用于高性能电脑运行。Codex 本地只做代码检查和 dry-run smoke test。

## A. 必补 HGCR 主实验数据

### 1. 补 low/high arrival 的 seed1/2

```powershell
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 1 --arrival_intensity low --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 2 --arrival_intensity low --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 1 --arrival_intensity high --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 2 --arrival_intensity high --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
```

### 2. 补 3×3 热力图缺失场景（seed0）

```powershell
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity low --carryover_ratio low --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity low --carryover_ratio high --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity medium --carryover_ratio low --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity medium --carryover_ratio high --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity high --carryover_ratio low --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity high --carryover_ratio high --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
```

### 3. 补敏感度点（seed0, medium-medium）

```powershell
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity medium --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 0.1 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity medium --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 2.0 --baseline_method fifo
```

## B. 多基线动态评估

```powershell
python evaluate_stage_G_dynamic_baselines.py --methods Random SPT LPT FIFO GreedyECT Lookahead MinLoad MLP-Ranker HGCR-PPO --save_schedule_trace
```

输出到：

```text
data/results/stage_G/paper_results/stage_G_method_comparison_detail__*.csv
data/results/stage_G/paper_results/schedule_trace__*.csv
```

## C. 汇总

```powershell
python summarize_stage_G_paper_results.py
python summarize_stage_G_paper_results_v2.py
```

## D. 画图

```powershell
python plot_stage_G_paper_figures.py
python plot_stage_G_paper_figures_v2.py
```

## E. 甘特图

```powershell
python generate_stage_G_gantt_cases.py
python generate_stage_G_gantt_cases_v2.py
```

## 本地轻量自检

```powershell
python -m py_compile run_hgcr_dynamic_ppo.py
python -m py_compile evaluate_stage_G_dynamic_baselines.py
python -m py_compile summarize_stage_G_paper_results.py
python -m py_compile summarize_stage_G_paper_results_v2.py
python -m py_compile plot_stage_G_paper_figures.py
python -m py_compile plot_stage_G_paper_figures_v2.py
python -m py_compile generate_stage_G_gantt_cases.py
python -m py_compile generate_stage_G_gantt_cases_v2.py

python evaluate_stage_G_dynamic_baselines.py --max_runs 1 --max_instances 1 --methods FIFO HGCR-PPO --dry_run --no_write
python summarize_stage_G_paper_results.py --dry_run --no_write
python summarize_stage_G_paper_results_v2.py --dry_run --no_write
python plot_stage_G_paper_figures.py --dry_run --no_write
python plot_stage_G_paper_figures_v2.py --dry_run --no_write
python generate_stage_G_gantt_cases.py --dry_run --no_write
python generate_stage_G_gantt_cases_v2.py --dry_run --no_write
```
