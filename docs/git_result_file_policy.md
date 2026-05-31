# Git Result File Policy

本仓库只提交代码、配置、固定算例管理逻辑和必要文档。实验运行产生的大量结果文件不提交到 GitHub。

## 基本规则

1. 代码文件、实验脚本、轻量文档和明确需要版本管理的配置应正常提交。
2. 实验结果文件不提交，包括 `.csv`、`.png`、`.zip`、`.xlsx`、`.log`、模型权重和 checkpoint 文件。
3. 远程高性能电脑运行实验后，输出到 `data/results/`、`outputs/`、`logs/`、`checkpoints/` 等目录的结果会保留在该电脑本地，并被 `.gitignore` 忽略。
4. 如果需要把实验结果发给他人分析，请手动打包上传或通过其他方式传输，不要把大批结果文件 commit 到 GitHub。
5. 如果某个小型结果表确实需要纳入论文材料，请复制到 `docs/` 或 `paper_assets/`，使用明确文件名并单独说明用途，不要直接提交 `data/results/` 下的大量实验输出。

## 目录占位

以下目录使用 `.gitkeep` 保留空目录结构：

- `data/results/`
- `outputs/`
- `logs/`
- `checkpoints/`

这些 `.gitkeep` 文件可以提交；目录中的实验产物仍然会被忽略。
