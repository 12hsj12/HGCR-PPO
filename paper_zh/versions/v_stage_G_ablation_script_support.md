# v_stage_G_ablation_script_support

## 本次范围

本次仅补齐 Stage G 消融实验的脚本支持和远端运行说明，不运行正式训练、不运行大规模评估、不生成正式实验结果。

## 脚本变更

1. `run_hgcr_dynamic_ppo.py`
   - 增加奖励组成消融入口：`util_only`、`cmax_only`、`util_plus_cmax`。
   - 增加动作规则库消融入口：通过 `--disabled_actions` 屏蔽指定内部规则。
   - 增加 `--ablation_family`、`--ablation_name`、`--no_fifo_outputs` 元数据和 no-FIFO 输出控制。
   - PPO 采样、回放评估和更新阶段共享同一动作 mask。

2. `evaluate_stage_G_dynamic_baselines.py`
   - HGCR-PPO checkpoint 回放时读取 manifest 中的禁用动作配置，保证动作库消融回放口径一致。

3. `run_stage_G_ablation_batch.py`
   - 新增远端批处理入口，支持 reward component ablation 与 action library ablation。
   - 默认输出到 `data/results/stage_G/ablation/`。
   - 支持 `--dry_run` 仅打印命令。

4. `summarize_stage_G_ablation_results.py`
   - 新增消融结果汇总脚本。
   - 只读取已有远端结果，不生成实验结果。
   - 导出 no-FIFO、no-unknown 清洗 CSV，可复制到 `paper_zh/data_used/`。

5. `plot_stage_G_ablation_figures.py`
   - 新增消融图候选生成脚本。
   - 仅在汇总 CSV 已存在时出图。

6. `paper_zh/notes/remote_ablation_run_commands.md`
   - 新增远端运行命令说明。

## 消融配置

奖励组成消融：

- `util_only`
- `cmax_only`
- `util_plus_cmax`

动作规则库消融：

- `full`
- `without_arrival_order_rule`
- `without_greedyect_rule`
- `without_lookahead_rule`
- `without_mlp_ranker_rule`

主配置：

- `size=small`
- `arrival_intensity=medium`
- `carryover_ratio=medium`
- `top_k=5`
- `episodes=5000`
- `reward_beta=5.0`
- `seeds=0,1,2`

## 输出约定

远端训练结果：

- `data/results/stage_G/ablation/runs/`

清洗 CSV：

- `data/results/stage_G/ablation/summaries/stage_G_reward_component_ablation_detail_no_fifo.csv`
- `data/results/stage_G/ablation/summaries/stage_G_reward_component_ablation_summary_no_fifo.csv`
- `data/results/stage_G/ablation/summaries/stage_G_action_library_ablation_detail_no_fifo.csv`
- `data/results/stage_G/ablation/summaries/stage_G_action_library_ablation_summary_no_fifo.csv`
- `paper_zh/data_used/stage_G_reward_component_ablation_detail_no_fifo.csv`
- `paper_zh/data_used/stage_G_reward_component_ablation_summary_no_fifo.csv`
- `paper_zh/data_used/stage_G_action_library_ablation_detail_no_fifo.csv`
- `paper_zh/data_used/stage_G_action_library_ablation_summary_no_fifo.csv`

候选图：

- `paper_zh/figures/appendix/Fig_A5_reward_component_ablation_no_fifo.png`
- `paper_zh/figures/appendix/Fig_A6_action_library_ablation_no_fifo.png`

## 写作口径

- 外部结果、图表和 CSV 不引入 FIFO 基线。
- 内部到达顺序动作写为 `Arrival-order rule`。
- 不使用 `unknown` size。
- 当前提交不包含任何新增实验数值结论。
