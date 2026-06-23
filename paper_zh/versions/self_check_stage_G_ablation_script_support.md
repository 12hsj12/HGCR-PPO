# self_check_stage_G_ablation_script_support

## 自检结论

- 是否未运行正式训练：是。本地仅计划执行 `py_compile` 与批处理 `--dry_run`。
- 是否未生成正式实验结果：是。新增汇总与绘图脚本均要求已有远端结果，当前不编造结果。
- 是否 no-FIFO：是。新增批处理默认使用 `--no_fifo_outputs`，外部评估输出不写 FIFO 行或 FIFO 指标列；内部到达顺序动作输出为 `Arrival-order rule`。
- 是否 no-unknown：是。远端批处理固定 `size=small`；汇总脚本跳过 `size=unknown`。
- 是否支持 reward component ablation：是，支持 `util_only`、`cmax_only`、`util_plus_cmax`。
- 是否支持 action library ablation：是，支持 `full`、`without_arrival_order_rule`、`without_greedyect_rule`、`without_lookahead_rule`、`without_mlp_ranker_rule`。
- 是否生成远端运行命令：是，见 `paper_zh/notes/remote_ablation_run_commands.md`。
- 是否仅做 py_compile / dry_run：是，未执行正式训练或大规模评估。
- 是否未编造结果：是，版本说明和远端命令只列配置、路径和预期文件名，不写数值结论。

## 需人工确认

- 远端运行时 MLP-Ranker checkpoint `checkpoints/stage_C/mlp_ranker/small_topk5_soft_ce/best.pt` 是否存在。
- 远端是否使用 GPU：如无 GPU，可将命令中的 `--device cuda` 改为 `--device cpu`。
- 远端正式跑完后，再决定消融图进入正文还是附录。
