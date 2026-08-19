# MERFISH Cell Type Annotation 最终预测包

## 结论

本包采用当前项目中完整 OOF 表现最高、且已经生成测试集概率的 **Graph-Regularized Class-wise Logit Stacker**。

- 完整三折 cross-fit OOF Accuracy：**82.50%**
- Macro-F1：**79.62%**
- Weighted-F1：**81.87%**
- Log loss：**0.4847**
- 训练/推理设备：NVIDIA GeForce RTX 4060 Laptop GPU
- 最终预测数：5,000 个测试细胞

需要特别区分：历史结果中出现的 `82.60%`、`82.69%`、`82.90%`、`83.02%` 等数值属于单折 held-out、调参折或 major-head 指标，不是全体 5,000 个训练细胞上的完整 OOF。当前可复现的最高完整 OOF 是 **82.50%**，因此本包没有把单折分数误报为最终模型准确率。

## 模型结构

最终模型以 `external_primary_crossfit` 为 Current Anchor，在 logit 空间进行 class-wise stacking。四个修正专家为：

- `external_refonly`：来自 `other_model` 的 reference-only LightGBM。
- `gene_token`：大参考数据集重训的 gene-token encoder。
- `segment_mnn`：同 Segment 的 MNN residual encoder。
- `soft_slot_segment_center`：带 Segment 去中心化的 soft-slot encoder。

输出公式为：

```text
anchor_logit
+ class-wise(external_refonly, gene_token, segment_mnn, soft_slot_segment_center)
+ bounded class bias
```

同时使用仅由外部 reference cells 建立的 60 类相似度图，对相邻生物类别的 logit correction 加 Graph-Laplacian 正则。该约束用于降低稀有类上的任意修正，不使用比赛 OOF 标签构图。

最终平均 class-wise correction 权重：

| 专家 | 平均权重 |
|---|---:|
| external_refonly | -0.0089 |
| gene_token | 0.1140 |
| segment_mnn | 0.0496 |
| soft_slot_segment_center | 0.0579 |

`gene_token` 是最终修正中最强的新增信息源；MNN 和 Segment-centered soft slot 提供较小但互补的修正。`external_refonly` 的直接平均增量接近零，但它已经参与 Current Anchor 的外部参考融合与生物约束过程。

## 与 other_model 的变化和提升

`other_model` 原始高准确率方案是基于互斥外部 reference cells 的 reference-only LightGBM。接入项目后，先加入 reference-derived Segment mask 与 E/I hard constraint，再与项目内部模型进行严格 cross-fitted 融合。

| 阶段 | OOF Accuracy | Macro-F1 | Log loss | 相对前一阶段 |
|---|---:|---:|---:|---:|
| other_model `external_refonly` | 82.04% | 79.23% | 0.6441 | - |
| Current Anchor `external_primary_crossfit` | 82.30% | 79.09% | 0.5443 | +0.26 pp |
| 最终 Graph-Regularized Stacker | **82.50%** | **79.62%** | **0.4847** | +0.20 pp |

最终模型相对 `other_model external_refonly` 的 Accuracy 总提升为 **+0.46 个百分点**。相对 Current Anchor 的 Accuracy 增益较小，配对 bootstrap 95% CI 为 `[-0.24, +0.66] pp`，McNemar `p=0.4264`，因此不能宣称统计显著；不过它是现有严格完整 OOF 中最高的配置，并把 log loss 从 `0.5443` 降至 `0.4847`，概率质量提升更明确。

近期 Biology-aware Glia Head 的最佳探索结果为 82.44%，低于 82.50%，且没有对应的完整测试集概率，因此未用于本次最终预测，避免将探索性 OOF 后处理混入提交。

## 预测文件

正式预测位于：

```text
prediction/prediction.csv
```

格式沿用比赛仓库当前 `prediction/prediction.csv`：

```csv
Cell_ID,MERFISH_cell_type_annotation.y
2796212800954100068,DH_ex_Grpr
2707604000123100074,oligodendrocyte_1
```

生成前审计结果：

- 行数为 5,000，与 `data/meta_test.csv` 完全一致。
- Cell_ID 数为 5,000，且没有重复或空 ID。
- Cell_ID 顺序与 `meta_test.csv` 完全一致。
- 没有空预测。
- 所有预测标签都来自训练集的 60 个合法类别；本次测试预测实际覆盖 58 类。

`model/submission_model_raw.csv` 保留训练脚本原始的 `Cell_ID,CellType` schema，便于模型审计；正式提交应以 `prediction/prediction.csv` 为准。

## 比赛 README 要求对应

根据仓库比赛 README：

- 目标列是 `MERFISH_cell_type_annotation`，评分指标是 confusion matrix 的 overall accuracy。
- 预测需通过 GitHub、以队长账号提交，并在截止时间前进入比赛仓库。
- 优胜队伍还需提交代码。

本包因此同时包含 canonical prediction、核心训练代码快照、模型指标、OOF/测试概率、class-wise 权重和类相似度图。原始比赛说明快照位于 `docs/competition_README.md`。

## 包内容

| 路径 | 用途 |
|---|---|
| `prediction/prediction.csv` | 按仓库 canonical schema 生成的正式预测 |
| `model/submission_model_raw.csv` | 模型脚本原始预测 |
| `model/metrics.json` | 完整 OOF、三折和配对审计指标 |
| `model/oof_probabilities.csv` | 训练集严格 OOF 概率 |
| `model/test_probabilities.csv` | 测试集 60 类概率 |
| `model/classwise_weights.csv` | 最终各类别的专家权重 |
| `model/gate_coefficients.csv` | gate 参数；本模型使用 fixed gate |
| `model/reference_class_graph.npz` | 外部 reference-only 类相似度图 |
| `model/class_diagnostics_summary.json` | 类别级修正诊断 |
| `code/train_graph_regularized_logit_stacker.py` | 类图正则 wrapper |
| `code/train_confidence_gated_logit_stacker.py` | class-wise stacker 主训练代码 |
| `docs/competition_README.md` | 比赛 README 快照 |

## 原项目复现命令

代码快照保留在包内；完整输入概率和默认数据路径仍以原项目 `C:\Users\lizhi\Documents\ChatGPT\hackathon` 为基准。原配置要求 CUDA，CPU fallback 被显式禁用。

```powershell
cd C:\Users\lizhi\Documents\ChatGPT\hackathon
C:\conda\envs\d2l\python.exe src\train_graph_regularized_logit_stacker.py `
  --class-graph outputs\reference_class_similarity_graph\reference_class_graph.npz `
  --graph-lambda 0.05 `
  --output-dir outputs\graph_regularized_logit_stacker `
  --epochs 800 `
  --learning-rate 0.03 `
  --gate-mode fixed `
  --seed 42
```

本次未重新训练模型，而是使用已经完成严格 OOF 训练并保存的最终测试概率生成提交文件，避免无必要重训造成结果漂移。