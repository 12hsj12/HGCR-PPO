# 阶段 G 下一轮实验与论文结果包命令

本文件用于高性能电脑运行。Codex 本地只做脚本检查，不跑完整实验。

## A. 必补 HGCR-PPO 动态场景

补齐 low/high arrival 的 seed1/2：

```powershell
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 1 --arrival_intensity low --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 2 --arrival_intensity low --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 1 --arrival_intensity high --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 2 --arrival_intensity high --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
```

## B. 可选 heatmap 补充

补 carryover low/high 的 6 个场景，seed0：

```powershell
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity low --carryover_ratio low --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity low --carryover_ratio high --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity medium --carryover_ratio low --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity medium --carryover_ratio high --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity high --carryover_ratio low --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity high --carryover_ratio high --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
```

## C. 基线评估

```powershell
python evaluate_stage_G_dynamic_baselines.py --methods Random SPT LPT FIFO GreedyECT Lookahead MinLoad MLP-Ranker HGCR-PPO
```

## D. 汇总

```powershell
python summarize_stage_G_paper_results.py
```

## E. 画图

```powershell
python plot_stage_G_paper_figures.py
```

## F. 甘特图

```powershell
python generate_stage_G_gantt_cases.py
```

## 本地轻量检查命令

```powershell
python -m py_compile evaluate_stage_G_dynamic_baselines.py
python -m py_compile summarize_stage_G_paper_results.py
python -m py_compile plot_stage_G_paper_figures.py
python -m py_compile generate_stage_G_gantt_cases.py

python evaluate_stage_G_dynamic_baselines.py --max_runs 1 --max_instances 1 --methods FIFO HGCR-PPO --dry_run --no_write
python summarize_stage_G_paper_results.py --dry_run --no_write
python plot_stage_G_paper_figures.py --dry_run --no_write
python generate_stage_G_gantt_cases.py --dry_run --no_write
```
