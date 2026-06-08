# 阶段 E 动态滚动汇总与画图命令

本文件用于区分本地轻量检查和高性能电脑上的完整阶段 E 汇总、画图流程。

## 1. 本地只允许执行的轻量检查命令

本地电脑性能有限，只做语法检查和极小 smoke test。不要在本地完整扫描全部 run，不要生成完整 CSV/MD/PNG。

```powershell
python -m py_compile summarize_stage_E_dynamic.py
python -m py_compile plot_stage_E_dynamic.py

python summarize_stage_E_dynamic.py --dry_run --max_runs 1 --no_write
python plot_stage_E_dynamic.py --dry_run --max_plots 1 --no_write
```

## 2. 高性能电脑上完整汇总命令

完整汇总会扫描：

```text
data/results/stage_F/hgcr_dynamic_ppo/runs/*
```

并把阶段 E 汇总结果写入：

```text
data/results/stage_E_dynamic_summary/
```

运行命令：

```powershell
python summarize_stage_E_dynamic.py
```

## 3. 高性能电脑上完整画图命令

完整画图读取 `summarize_stage_E_dynamic.py` 生成的最新 CSV，并把图片写入：

```text
data/results/stage_E_dynamic_summary/figures/{timestamp}_{uuid}/
```

运行命令：

```powershell
python plot_stage_E_dynamic.py --summary_dir data/results/stage_E_dynamic_summary
```

## 4. 需要打包回传给 ChatGPT 的文件列表

```text
data/results/stage_E_dynamic_summary/stage_E_all_runs__*.csv
data/results/stage_E_dynamic_summary/stage_E_beta_ablation__*.csv
data/results/stage_E_dynamic_summary/stage_E_arrival_generalization__*.csv
data/results/stage_E_dynamic_summary/stage_E_action_ratio_summary__*.csv
data/results/stage_E_dynamic_summary/stage_E_dynamic_report__*.md
data/results/stage_E_dynamic_summary/figures/*/*.png
```
