# 阶段 G 动态滚动 HGCR-PPO 命令

阶段 G：动态滚动场景下的 HGCR-PPO 正式训练与实验。

新的阶段 G 动态训练结果写入：

```text
data/results/stage_G/hgcr_dynamic_ppo/runs/{run_id}/
```

阶段 G 汇总结果写入：

```text
data/results/stage_G/summary/
```

阶段 G 图表写入：

```text
data/results/stage_G/figures/{timestamp}_{uuid}/
```

## 1. 本地只允许执行的轻量检查命令

本地电脑性能有限，不允许启动训练、不允许完整汇总、不允许完整画图。

```powershell
python -m py_compile run_hgcr_dynamic_ppo.py
python -m py_compile summarize_stage_G_dynamic.py
python -m py_compile plot_stage_G_dynamic.py

python summarize_stage_G_dynamic.py --dry_run --max_runs 1 --no_write
python plot_stage_G_dynamic.py --dry_run --max_plots 1 --no_write
```

不要在本地运行：

```powershell
python summarize_stage_G_dynamic.py
python plot_stage_G_dynamic.py
```

## 2. 高性能电脑上重新跑阶段 G 干净数据的命令

### A. reward_beta 消融，medium-medium seed0

```powershell
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity medium --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 0.01 --baseline_method fifo

python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity medium --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 1.0 --baseline_method fifo

python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity medium --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
```

### B. seed 稳定性，medium-medium beta=5.0

medium-medium beta=5.0 seed0 已包含在 A 中，不需要重复。

```powershell
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 1 --arrival_intensity medium --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo

python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 2 --arrival_intensity medium --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
```

### C. arrival 泛化，beta=5.0 seed0

medium-medium beta=5.0 seed0 已包含在 A 中，不需要重复。

```powershell
python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity low --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo

python run_hgcr_dynamic_ppo.py --size small --top_k 5 --episodes 5000 --seed 0 --arrival_intensity high --carryover_ratio medium --reward_mode util_plus_cmax --reward_beta 5.0 --baseline_method fifo
```

## 3. 高性能电脑上完整汇总命令

```powershell
python summarize_stage_G_dynamic.py
```

## 4. 高性能电脑上完整画图命令

```powershell
python plot_stage_G_dynamic.py --summary_dir data/results/stage_G/summary
```

## 5. 需要打包回传给 ChatGPT 的文件列表

```text
data/results/stage_G/summary/stage_G_all_runs__*.csv
data/results/stage_G/summary/stage_G_beta_ablation__*.csv
data/results/stage_G/summary/stage_G_seed_stability__*.csv
data/results/stage_G/summary/stage_G_arrival_generalization__*.csv
data/results/stage_G/summary/stage_G_action_ratio_summary__*.csv
data/results/stage_G/summary/stage_G_dynamic_report__*.md
data/results/stage_G/figures/*/*.png
```
