# Stage F Reference Mapping: Conservative Rule-Selector PPO

本文档用于整理 HGCR-PPO 阶段 F 的外部参考项目与本项目实现边界。阶段 F 的目标不是复刻标准 JSSP/FJSP 代码，而是在本项目已有滚动调度环境、HybridTopK 候选集、MLP-Ranker 和结果保存机制上，实现一个保守的 Rule-Selector PPO。

## 1. 参考项目与可借鉴点

### 1.1 L2D: Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning

GitHub: https://github.com/zcaicaros/L2D

L2D 是 JSSP 上的逐步派工 DRL 框架，公开仓库包含 `JSSP_Env.py`、`PPO_jssp_multiInstances.py`、`agent_utils.py` 等模块。它的核心启发是把调度过程建模为连续决策过程：每一步选择一个当前可调度操作，环境推进，直到形成完整排程。

本项目可借鉴：

- 逐步 dispatch 的 MDP 建模方式。
- Actor-Critic / PPO / rollout / validation checkpoint 的训练结构。
- 学习一个派工策略，而不是直接求解完整数学规划模型。

本项目不能照搬：

- L2D 面向标准 JSSP，操作工序和机器路径固定；HGCR-PPO 面向钢铁加工中心的任务拆分、非等效并行产线、多周期滚动调度。
- L2D 的 job-level 或 operation-level 动作空间不适合直接复用到本项目，因为此前普通 job-level PPO 已经在泛化上失败。
- L2D 的图状态和基准实例格式不能替代本项目的固定 train/val/test 实例体系。

### 1.2 fjsp-drl: Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning

GitHub: https://github.com/songwenas12/fjsp-drl

该项目实现了 FJSP 的 GNN + DRL 框架，强调 operation/job 与 machine 的兼容关系建模，并在训练中处理可行动作约束。

本项目可借鉴：

- operation/job 与 machine 关系图的建模思想。
- DRL 调度中必须显式处理可行动作集合的原则。
- action mask 思路：不可用动作不能只靠负奖励惩罚，而应在采样分布中直接屏蔽。

本项目不能照搬：

- 阶段 F 暂不实现新的 GNN 或 operation-machine 双图 encoder。
- 标准 FJSP 的 operation-machine 选择不能直接映射到本项目的“任务拆分 + 多产线滚动调度”。
- 本项目已有阶段 D 的 GNN-Ranker 结果，阶段 F 当前只做 DRL 模块消融第一版。

### 1.3 End-to-end-DRL-for-FJSP

GitHub: https://github.com/Lei-Kun/End-to-end-DRL-for-FJSP

该方向把 FJSP 拆成多动作决策，例如 operation selection 与 machine assignment 的联合或分解决策。它说明 FJSP 不一定必须用单一巨大动作空间建模。

本项目可借鉴：

- 多动作或分解动作的建模思想。
- 将复杂调度动作拆为更稳定的上层选择与下层执行。
- 为后续扩展“任务选择 + 拆分数量”联合策略留下接口。

本项目不能照搬：

- 阶段 F 不做 operation selection / machine assignment 的完整端到端复现。
- 当前环境的 machine assignment 与 split ratio 已由 `RollingSchedulingEnv` 和 `choose_split_num` 处理，PPO 不直接重写这些底层逻辑。
- 当前只需要 Rule-Selector PPO，不做复杂 PPO+GNN 或多头任务拆分策略。

### 1.4 FJSP-DRL / DANIEL

GitHub: https://github.com/wrqccc/FJSP-DRL

DANIEL 类方法关注 operation-machine 双注意力结构，用注意力机制提取 FJSP 中操作与机器之间的适配关系。

本项目可借鉴：

- 机器适配关系是调度状态的重要组成部分。
- 后续若重启图结构增强，可参考双注意力或双部图关系建模。
- 对候选动作做结构化表示，而不是只使用全局标量。

本项目不能照搬：

- 阶段 F 不新增 DANIEL 式 attention encoder。
- 当前 GNN-Ranker 尚未超过 MLP-Ranker，直接叠加复杂图结构不符合阶段 F 的保守目标。
- 本项目的动作是 rule_id，不是 operation-machine pair。

### 1.5 PPO selecting dispatching rules for dynamic/random-arrival FJSP 的思想

该类思路与阶段 F 最贴合：PPO 不直接从大量 job 中选择，而是在成熟 dispatching rules 中选择一个 rule，再由 rule 在当前可行动作集合中完成具体 job 选择。

本项目可借鉴：

- 将 PPO 动作空间定义为规则集合，而不是 job 集合。
- 使用成熟启发式降低探索风险。
- 对动态到达、滚动调度、候选集变化的环境更友好。
- 通过 rule distribution 观察 DRL 是否真的学习到非 FIFO 的决策模式。

本项目不能照搬：

- 不直接套用外部 FJSP 的 reward、状态特征或 benchmark 格式。
- 不让 PPO 绕开本项目已有 HybridTopK、MLP-Ranker、schedule_validator 和固定数据集。
- 不把 OracleDebug 当作论文主方法，它只用于诊断上限。

## 2. HGCR-PPO 与标准 JSSP/FJSP 的差异

HGCR-PPO 的研究对象不是标准静态 JSSP/FJSP，而是面向钢铁加工中心的任务拆分、非等效并行产线、多周期滚动调度问题。主要差异如下：

- 标准 JSSP 通常有固定工序顺序和固定机器路径；本项目任务可在候选产线中选择，并可能拆分到多台候选机器。
- 标准 FJSP 关注 operation-machine assignment；本项目同时关注 release time、rolling period、candidate machines、split number、machine availability 和全过程 Cmax_roll。
- 本项目固定评价体系已经建立，所有算法必须在同一批 fixed train/val/test instances 上比较。
- 本项目已有 HybridTopK 和 Ranker 成果，阶段 F 应复用这些结构，而不是回到裸 job-level PPO。

## 3. 为什么采用 Rule-Selector PPO 而不是 job-level PPO

阶段二已经表明普通 MLP-PPO 虽可运行，但泛化能力不足；BC-only 有帮助但仍弱于强启发式；PPO fine-tuning 容易破坏 BC 策略。继续在 job-level 动作空间上调 PPO 超参数收益很低。

阶段 F 采用 Conservative Rule-Selector PPO 的理由：

- 动作空间稳定：动作是 `rule_id`，初始规则为 `fifo`、`lookahead`、`greedy_ect`、`minload`、`mlp_ranker_soft_ce`，可选增加 `mlp_ranker_pairwise`。
- 执行仍保守：每个 rule 只在 HybridTopK 候选集中推荐 job，避免 PPO 在全 job 集合中盲目探索。
- 可解释性更强：可以统计每种 rule 的选择比例，直接观察 DRL 模块是否学到何时信任启发式或 Ranker。
- 兼容已有成果：保留 `RollingSchedulingEnv.step((job_id, split_num))`、HybridTopK、MLP-Ranker、schedule_validator、CSV 聚合机制。
- 风险更低：action mask 屏蔽不可用规则，例如未提供 MLP checkpoint 时自动屏蔽对应 ranker rule。

## 4. 阶段 F 实现边界

当前阶段只实现 DRL 模块消融第一版：

- FIFO
- MLP-Ranker
- Rule-Selector PPO
- Oracle Debug

当前阶段不实现：

- 新 GNN。
- GNN 消融。
- 拆分策略消融。
- PPO+GNN 复杂联合模型。
- medium/large 完整训练或完整评估。

阶段 F 的代码自检只运行轻量命令：

```bash
python -m py_compile rule_selector_env.py
python -m py_compile train_rule_selector_ppo.py
python -m py_compile evaluate_rule_selector_ppo.py
```

