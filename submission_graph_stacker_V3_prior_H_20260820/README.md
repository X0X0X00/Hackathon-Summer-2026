# Prior-H Graph Stacker V3

## 最终结论

- 最终提交模型：`depth_masked_prior_h_anchor`
- 选择规则：The newly trained stacker did not beat the Prior-H Anchor, so submission falls back to the safer OOF winner.
- OOF Accuracy：`82.66%`
- OOF Macro-F1：`79.7739%`
- OOF Weighted-F1：`81.9008%`
- OOF Log loss：`0.541130`
- 训练设备：`NVIDIA GeForce RTX 4060 Laptop GPU`
- 测试集启用 strict H：`634 / 5000` 个细胞（`12.68%`）

`prediction/prediction.csv` 是最终提交文件。测试集没有标签，因此所有模型选择只依据三折 OOF，不依据 test 预测分布。

## 本次吸收了什么

最终主干不是把所有实验粗暴平均，而是分成两层：

1. `Prior-H Anchor`：先用保守生物先验形成 CurrentAnchor；只有高深度、可靠基因足够、严格 MNN 图连接足够的细胞才注入 strict H。低深度细胞的 H 权重严格为 0。
2. `Graph-Regularized Class-wise Stacker`：围绕 Prior-H Anchor 学习每个细胞类型的专家修正，并用外部参考细胞构造的类别图约束相近类别的修正不要剧烈分叉。

最终 stacker 输入四个非 `other_model` 概率 head：

| Head | 在最终模型中的作用 |
|---|---|
| V2 graph stacker | 提供历史模型之间已经验证过的类别级互补与更好的概率校准 |
| CurrentAnchor + biology priors | 提供 Segment/EI/AP、胶质、解剖拓扑和有序少突成熟先验 |
| Gene-token encoder | 提供基因 token 之间的非线性交互和低 log-loss 证据 |
| Strict invariant H | 提供同 Segment 生物邻居上下文，允许 stacker 对特定类别做额外修正 |

## 最终指标

| 模型 | OOF Accuracy | Macro-F1 | Weighted-F1 | Log loss |
|---|---:|---:|---:|---:|
| Prior-H Anchor | 82.66% | 79.7739% | 81.9008% | 0.541130 |
| 本次训练 graph stacker | 82.54% | 79.9138% | 81.9400% | 0.483225 |
| 最终自动选择 | 82.66% | 79.7739% | 81.9008% | 0.541130 |

## 各 head 的代码、功能、原理和单独成绩

以下均为我们自己的 head 或融合代码。`other_model` 只作为外部概率/缓存数据源，没有复制进本包。

| Head | 代码文件 | 功能与原理 | 单独 OOF 数据 |
|---|---|---|---:|
| External primary Anchor | 外部基线，仅作为比较项 | 高准确率树模型/外部参考融合，是所有修正的稳定锚点 | Accuracy 82.30%，Macro-F1 79.0876%，Log loss 0.544314 |
| Gene-token encoder | `code/heads/train_external_gene_token_encoder.py` | 把每个基因作为 token，学习基因间非线性交互；不依赖空间近邻 | Accuracy 81.46%，Macro-F1 78.1222%，Log loss 0.488085 |
| Segment/EI/AP prior | `code/heads/evaluate_current_anchor_metadata_prior_head.py` | 交叉拟合的分层 empirical-Bayes 类别先验；以 log-prior residual 修正 Anchor | Head-only 46.16%；严格 nested 融合约 82.30%，说明信息大多已被 Anchor 吸收 |
| Biology-aware glia head | `code/heads/train_biology_aware_glia_head.py` | 用胶质家族 marker、竞争类别和零表达否决约束高置信修正 | 最佳探索 Accuracy 82.40%，Macro-F1 79.5084% |
| Missing-marker residual | `code/heads/evaluate_anchor_marker_completion_residual.py` | 仅对高预测力缺失 marker 用 MNN 微弱补全，避免把伪缺失当真零 | 最佳 Accuracy 82.30%；未超过 Anchor，最终只保留为弱证据 |
| Anatomical topology prior | `code/heads/evaluate_anatomical_topology_prior.py` | 把 Segment/AP 的有序解剖连续性作为小残差，而不是硬编码类别 | 最佳 Accuracy 82.32% |
| Ordered oligodendrocyte prior | `code/heads/evaluate_ordered_oligodendrocyte_prior.py` | 对 OPC、progenitor、成熟少突建立有序成熟轴，限制不合理跨级修正 | Accuracy 82.44%，Macro-F1 79.1012% |
| Conservative biology union | `code/heads/audit_ordered_biology_head_overlap.py`、`code/heads/export_biology_head_test_and_combine.py` | 合并胶质、少突和拓扑先验，只接受交叉验证支持且冲突可控的修正 | Accuracy 82.54%，Macro-F1 79.4508%，Log loss 0.543322 |
| Legacy Segment MNN H | `code/heads/build_external_segment_mnn_graph.py`、`code/heads/train_external_mnn_residual_encoder.py` | 同 Segment 双向互近邻；H 为邻居表达减 Segment 均值 | H Accuracy 79.44%；与旧 Anchor 融合 82.20% |
| Soft-slot Segment H | `code/heads/build_soft_slot_neighbor_graph.py`、`code/heads/train_external_mnn_residual_encoder.py` | Slot-wise 竞争选择邻居，降低固定距离规则偏差 | H Accuracy 79.16%；与旧 Anchor 最好 82.24% |
| Strict invariant Piecewise H | `code/heads/run_invariant_mnn_gonogo.py`、`code/heads/run_reliability_piecewise_soft_slot_gonogo.py`、`code/heads/build_invariant_piecewise_segment_mnn_graphs.py`、`code/heads/train_invariant_piecewise_mnn_segment_centered_encoder.py` | 同 Segment MNN；邻居选择同时使用 missingness-invariant 表示、可靠零证据和分档表达；聚合时减全 Segment 平均表达 | H Accuracy 78.74%，Macro-F1 75.6560%，Log loss 0.622455；全局融合 82.48% |
| Depth-masked Prior-H Anchor | `code/heads/run_depth_masked_strict_h_gonogo.py` | 仅当 `n_detected > 14`、可靠基因 `>=20`、严格图度数 `>=4` 时注入 21% H；否则完全使用 CurrentAnchor | Accuracy 82.66%，Macro-F1 79.7739%，Weighted-F1 81.9008%，Log loss 0.541130；相对 CurrentAnchor +6/-0 |
| V2 graph stacker | `code/heads/train_confidence_gated_logit_stacker.py`、`code/heads/train_graph_regularized_logit_stacker.py` | 类别级 logit stacking + 参考类别图 Laplacian 正则 | Accuracy 82.50%，Macro-F1 79.6233%，Log loss 0.484742 |
| V3 Prior-H graph stacker | `code/train_final_prior_h_graph_stacker.py` | 用 Prior-H 作锚点，联合 V2、prior、gene-token、strict H，并继续施加类别图正则 | Accuracy 82.54%，Macro-F1 79.9138%，Log loss 0.483225 |

## H 的生物学含义

H 不是一般空间平均。它执行四重限制：

1. 邻居必须在同一 `Segment`，保证脊髓大体解剖方向不走偏。
2. 邻居必须双向互近，降低单向近邻和大簇吸附。
3. 邻居检索同时使用 missingness-invariant、可靠零证据和 Piecewise 分档表达，避免把技术性零当作生物跃迁。
4. 聚合表达减去该 Segment 全部细胞的平均表达谱，H 表示的是组内偏离，而不是重复编码 Segment 元数据。

深度门控是必要的。严格 H 单独能力强，但低深度细胞中的缺失会制造伪离散；因此低深度样本不使用 H，而不是交给表现明显较差的纯树路由。

## 一键 GPU 运行

硬件和环境：

- GPU：NVIDIA RTX 4060 Laptop GPU 或其他 CUDA GPU
- Python：`C:\conda\envs\d2l\python.exe`
- 默认项目：`C:\Users\lizhi\Documents\ChatGPT\hackathon`
- 默认大数据缓存：`C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token`

在 PowerShell 中运行：

```powershell
& ".\run_all.ps1"
```

指定其他项目或 Python 路径：

```powershell
& ".\run_all.ps1" `
  -ProjectRoot "C:\path\to\hackathon" `
  -CacheDir "C:\path\to\gene_token" `
  -Python "C:\conda\envs\d2l\python.exe"
```

流水线顺序：生成 test Prior-H 路由，训练 800 epoch 三折 graph stacker，比较 stacker 与 Prior-H Anchor，导出最终预测并重建 README。

## 目录说明

| 路径 | 内容 |
|---|---|
| `prediction/prediction.csv` | 最终 5000 行提交文件 |
| `model/final_selection.json` | 最终模型选择、OOF 指标和 test 路由统计 |
| `model/oof_probabilities_final.csv` | 最终选中模型的 OOF 概率 |
| `model/test_probabilities_final.csv` | 最终选中模型的 test 概率 |
| `model/inputs/` | 五个 stacker 输入 head 及 Prior-H 路由审计 |
| `model/trained_stacker/` | 本次训练的权重、gate、OOF/test 概率和 metrics |
| `model/audits/` | 各先验与 H 实验的原始指标快照 |
| `code/heads/` | 所有非 `other_model` head 的源码快照 |
| `run_all.ps1` | 一键 GPU 复现入口 |

## 提交格式

`prediction/prediction.csv` 两列：

```text
Cell_ID,MERFISH_cell_type_annotation.y
```

共 5000 个测试细胞，每个 `Cell_ID` 恰好一行。
