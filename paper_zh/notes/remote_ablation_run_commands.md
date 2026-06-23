# Stage G 消融实验远端运行命令

本文档用于远端电脑补跑 Stage G no-FIFO、no-unknown 消融实验。当前提交只提供脚本支持，不包含正式实验结果。

## 1. 更新代码

```powershell
cd E:\article
git pull origin HGCR-PPO-1.7.0-G
```

## 2. 先检查计划命令

```powershell
python run_stage_G_ablation_batch.py --only all --dry_run
```

## 3. 奖励组成消融

主配置：`size=small`、`arrival=medium`、`carryover=medium`、`top_k=5`、`episodes=5000`、`reward_beta=5.0`、`seeds=0 1 2`。

```powershell
python run_stage_G_ablation_batch.py --only reward --device cuda
```

包含：

- `util_only`
- `cmax_only`
- `util_plus_cmax`

## 4. 动作规则库消融

主配置：`size=small`、`arrival=medium`、`carryover=medium`、`top_k=5`、`episodes=5000`、`reward_mode=util_plus_cmax`、`reward_beta=5.0`、`seeds=0 1 2`。

```powershell
python run_stage_G_ablation_batch.py --only action --device cuda
```

包含：

- `full`
- `without_arrival_order_rule`
- `without_greedyect_rule`
- `without_lookahead_rule`
- `without_mlp_ranker_rule`

内部到达顺序动作在输出中统一写为 `Arrival-order rule`，不作为外部 FIFO 基线。

## 5. 汇总清洗结果

```powershell
python summarize_stage_G_ablation_results.py --export_paper
```

预计输出：

- `data/results/stage_G/ablation/summaries/stage_G_reward_component_ablation_detail_no_fifo.csv`
- `data/results/stage_G/ablation/summaries/stage_G_reward_component_ablation_summary_no_fifo.csv`
- `data/results/stage_G/ablation/summaries/stage_G_action_library_ablation_detail_no_fifo.csv`
- `data/results/stage_G/ablation/summaries/stage_G_action_library_ablation_summary_no_fifo.csv`
- `paper_zh/data_used/stage_G_reward_component_ablation_detail_no_fifo.csv`
- `paper_zh/data_used/stage_G_reward_component_ablation_summary_no_fifo.csv`
- `paper_zh/data_used/stage_G_action_library_ablation_detail_no_fifo.csv`
- `paper_zh/data_used/stage_G_action_library_ablation_summary_no_fifo.csv`

## 6. 生成论文候选图

```powershell
python plot_stage_G_ablation_figures.py
```

预计输出：

- `paper_zh/figures/appendix/Fig_A5_reward_component_ablation_no_fifo.png`
- `paper_zh/figures/appendix/Fig_A6_action_library_ablation_no_fifo.png`

## 7. 注意事项

- 不使用 `size=unknown`。
- 输出结果不包含外部 FIFO 对比。
- 如 MLP-Ranker checkpoint 缺失，应在运行日志中记录，不得写成算法贡献或实验结论。
- 运行完成前不要在论文正文中写入数值结论。
