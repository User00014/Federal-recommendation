# Federated_Privacy_Project 目录整理说明

当前目录按用途建议这样理解：

- `src/`：核心实验代码。
- `data/`：原始数据集与说明。
- `logs/`、`最终数据/`、`备用数据/`：本地结果与对照数据。
- `cloud_results/`：云端拉回结果。
- `cloud_results/archive_intermediate_snapshots_20260311/`：已归档的中间快照、运行时抓取和应急拉回目录。
- `figures/`：生成后的图表。
- `figures/archive_exploratory_20260309/`：2026-03-09 生成的探索性图表归档。
- `reports/`：阶段/最终实验报告 Markdown。
- `output/`：结构图和 PPT 导出物。

约定：

- `final_pull_*`、`pull_seed52_*` 这类仍被报告脚本直接使用，保留在 `cloud_results/` 根目录。
- 中间快照统一收进 `archive_intermediate_snapshots_20260311/`，避免根目录继续堆积。
- 重新生成报告时，运行 `generate_stage_report.py` 或 `generate_final_report.py` 会自动输出到 `reports/`。
- 历史探索图默认输出到 `figures/archive_exploratory_20260309/`，不再和正式报告图混放。
