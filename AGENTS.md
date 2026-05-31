# AGENTS.md

本文件用于指导 Codex 在当前仓库中继续开发 HGCR-PPO 项目。

重要说明：本文件为中文内容，文件编码为 UTF-8 with BOM。请不要将本文件另存为 ANSI、GBK 或 UTF-16。若在 Windows 中编辑，请使用 VS Code，并确认右下角编码为 UTF-8。

---

## 1. 项目定位

当前仓库名称已经由 TSG-PPO 修改为 HGCR-PPO。

HGCR-PPO 的全称为：

Heuristic-guided Graph Candidate Ranking Proximal Policy Optimization

中文名称为：

启发式引导的图候选排序近端策略优化算法。

本项目研究对象是：

面向钢铁加工中心的任务拆分、非等效并行产线、多周期滚动调度问题。

优化目标是：

最小化滚动调度全过程的实际最大完工时间 Cmax_roll。

---

## 2. 当前研究背景

原项目已经完成以下基础内容：

1. 算例生成器。
2. 滚动调度环境。
3. 启发式基线。
4. 评价指标计算。
5. 甘特图输出。
6. small、medium、large 三种规模算例运行。

阶段二已经充分尝试普通 MLP-PPO、PPO-Order、BC 预训练、freeze actor、小学习率、小 clip ratio、少 update epoch 等方法。

当前阶段性结论是：

普通 MLP-PPO 工程上可以运行，但泛化能力不足。BC-only 有一定帮助，但仍弱于 FIFO、GreedyECT 等启发式。PPO fine-tuning 会破坏 BC 策略。继续单纯调整 PPO 超参数收益很低。

因此，后续主线不再是继续盲目调普通 PPO，而是转向：

固定评价体系
-> 拆分价值测试
-> 强启发式候选集
-> 多专家排序学习
-> GNN-Ranker
-> Conservative PPO 微调

---

## 3. 当前阶段目标

当前处于阶段 A：重建稳定、可复现、可消融的实验评价体系。

阶段 A 的目标不是训练新模型，也不是实现 GNN 或 PPO。

阶段 A 只做以下事情：

1. 固定 train、val、test 数据集。
2. 统一评估入口。
3. 统一结果 CSV 格式。
4. 复测已有启发式基线。
5. 进行拆分价值测试。
6. 保存用于论文后续实验的稳定基础结果。

阶段 A 不要做以下事情：

1. 不要实现 GNN。
2. 不要实现 Ranker。
3. 不要实现 Conservative PPO。
4. 不要继续调普通 PPO 超参数。
5. 不要重写已有调度环境。
6. 不要删除已有核心文件。

---

## 4. 必须保留的原有文件

除非明确要求，否则不要删除或大改以下文件：

1. run_baselines.py
2. run_ppo.py
3. rolling_scheduling_env.py
4. heuristics.py
5. metrics.py
6. visualization.py
7. instance_generator.py
8. data/results 中已有历史结果

如果需要修改这些文件，必须保持向后兼容。

尤其要保证：

python run_baselines.py

仍然可以运行。

---

## 5. 阶段 A 推荐新增文件

请优先新增以下文件，而不是推翻已有结构：

1. instance_manager.py
   用于生成、保存、加载固定 train、val、test 数据集。

2. evaluate_methods.py
   用于统一评估启发式和后续算法。

3. run_stage_A.py
   用于一键运行阶段 A。

4. check_split_effect.py
   用于测试任务拆分策略对 Cmax 的影响。

5. experiment_registry.py
   可选，用于统一登记 method、candidate_mode、split_rule 等实验配置。

---

## 6. 固定数据集要求

固定数据集保存路径建议为：

data/instances/fixed/train/
data/instances/fixed/val/
data/instances/fixed/test/

每个规模生成：

small:
train 200
val 30
test 50

medium:
train 200
val 30
test 50

large:
train 200
val 30
test 50

每个实例至少包含：

1. instance_id
2. size
3. seed
4. jobs
5. machines
6. process_type
7. release_time
8. candidate_machines
9. processing_time
10. max_split_num
11. rolling_period_length
12. num_periods

要求：

1. 使用固定 random seed。
2. 所有算法必须在同一批 test instances 上比较。
3. 评估阶段不得临时重新生成实例。
4. 生成逻辑优先复用 instance_generator.py。

---

## 7. 统一评估要求

evaluate_methods.py 需要支持：

python evaluate_methods.py --size small --split test
python evaluate_methods.py --size medium --split test
python evaluate_methods.py --size large --split test

当前阶段至少评估以下方法：

1. Random
2. FIFO
3. SPT
4. LPT
5. MinCandidateLoad
6. GreedyECT

每个方法在固定 test instances 上运行，输出每个实例的结果，并生成 summary。

结果保存到：

data/results/stage_A/stage_A_baselines.csv
data/results/stage_A/stage_A_summary.csv

单实例 CSV 字段至少包括：

1. method
2. size
3. split
4. seed
5. instance_id
6. Cmax_roll
7. average_completion_time
8. average_waiting_time
9. machine_utilization
10. load_balance_std
11. split_task_ratio
12. total_split_count
13. inference_time
14. candidate_mode
15. split_rule
16. notes

summary CSV 需要按 method 和 size 聚合 mean 和 std。

---

## 8. 阶段 A 一键运行

run_stage_A.py 需要支持：

python run_stage_A.py --sizes small
python run_stage_A.py --sizes small medium large

功能：

1. 检查 fixed dataset 是否存在。
2. 若不存在，则自动生成。
3. 调用 evaluate_methods.py 评估启发式。
4. 保存 stage_A_baselines.csv。
5. 保存 stage_A_summary.csv。
6. 每个规模至少保存 1 张 FIFO 甘特图和 1 张 GreedyECT 甘特图。

结果目录：

data/results/stage_A/

---

## 9. 拆分价值测试

check_split_effect.py 用于判断当前问题中任务拆分是否真的能改善 Cmax。

运行方式：

python check_split_effect.py --size small --split test
python check_split_effect.py --size medium --split test
python check_split_effect.py --size large --split test

固定任务排序方式至少支持：

1. FIFO ordering
2. GreedyECT ordering

在固定排序下测试以下拆分策略：

1. NoSplit：所有任务 split_num = 1。
2. MaxSplit：所有任务使用最大可行拆分数。
3. EqualSplit：多产线均分。
4. SpeedRatioSplit：按加工速度反比分配。
5. GreedyECTSplit：当前 GreedyECT 拆分规则。
6. RandomSplit：随机拆分数量。
7. OracleSplitDebug：枚举局部可行 split_num，选择当前局部 Cmax 最小的 split_num。

输出文件：

data/results/stage_A/split_effect_summary.csv

注意：

OracleSplitDebug 只用于诊断拆分上限，不作为最终论文主方法。

---

## 10. 消融实验同步推进原则

后续所有开发必须注意消融实验同步推进。

但不要额外增加无关消融。

此前规划中的消融实验已经足够支撑论文完成，包括：

1. 候选集消融。
2. 学习方式消融。
3. 图结构消融。
4. 拆分策略消融。
5. PPO 微调消融。

当前阶段 A 只需要推进：

拆分价值测试。

这一步是后续拆分策略消融的基础。

不要在阶段 A 额外实现候选集消融、GNN 消融或 PPO 消融。

---

## 11. 后续阶段方向，仅作为上下文

阶段 B：
增强强启发式和候选集，包括 Lookahead Greedy、Beam Search、Hybrid TopK。

阶段 C：
实现 Multi-expert BC、MLP-Ranker、候选集排序学习。

阶段 D：
实现任务-产线二部图、GNN-Ranker、图结构消融。

阶段 E：
在 GNN-Ranker 基础上实现 Conservative PPO 微调，并进行 PPO 微调消融。

当前只做阶段 A。

---

## 12. 代码质量要求

1. 不要写成一次性 demo。
2. 所有新增模块应便于后续扩展。
3. 统一保存结果，避免散落到多个目录。
4. 保证命令行参数清晰。
5. 保证随机种子可控。
6. 保证已有 baseline 可复现。
7. 关键函数要有必要注释。
8. 不要添加大量无意义注释。
9. 不要引入过重依赖。
10. 不要破坏已有项目结构。

---

## 13. 完成任务后需要汇报

每次完成修改后，请说明：

1. 新增了哪些文件。
2. 修改了哪些原文件。
3. 如何运行。
4. 结果保存在哪里。
5. 是否保持 run_baselines.py 兼容。
6. 当前阶段对应哪个消融基础。
7. 下一步建议是什么。

---

## 14. 编码要求

本文件必须保存为 UTF-8 with BOM。

如果在 Windows PowerShell 中重写本文件，建议使用：

Set-Content -Path AGENTS.md -Value $content -Encoding utf8BOM

如果使用 VS Code，请确认右下角编码显示为 UTF-8，并优先选择 Save with Encoding -> UTF-8 with BOM。

不要保存为 ANSI、GBK 或 UTF-16。
