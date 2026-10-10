# Ecom Search Lab

面向普通 CPU 的电商搜索研究项目：查询理解、词法与向量召回、CrossEncoder 精排、LambdaMART 融合，以及独立的图文检索和个性化实验。

基于 [Amazon ESCI](https://github.com/amazon-science/esci-data) 的公开数据与研究代码扩展，用于学习、实验和项目展示。与 TikTok/字节跳动无隶属关系。

## 从真实数据运行完整实验

需要 Python 3.10+，不需要 GPU。独立公开入口从固定提交的官方 ESCI Parquet 开始，生成查询划分、训练模型、封存测试并启动浏览器搜索。默认数据配置已实跑：43,222 商品，1,000/100/100 个训练/验证/测试查询，全部使用官方人工标注。

```bash
git clone https://github.com/chendi-Shi/ecom-search-lab.git
cd ecom-search-lab
python -m pip install -r requirements-public.txt
python scripts/prepare_public_esci.py --download
python scripts/benchmark_public.py select
python scripts/benchmark_public.py final --per-query
python scripts/serve_public.py --port 18867
```

打开 <http://127.0.0.1:18867/>，比较同一候选池的词法顺序与学习排序，检查 12 个特征的加权贡献、当前模型指纹和实际评测结果。`select` 只解析训练与验证标签；`final` 校验数据、源码和模型指纹后才打开测试标签，并拒绝同目录的重复最终评测。

首次下载约 1.16 GB 官方原始数据，导入器逐批扫描并校验完整 SHA256。已有相同原始数据时用 `--raw <directory>` 代替 `--download`。PyArrow 只用于导入 Parquet；该基准的训练、评测与服务使用 Python 标准库。请从源码仓库根目录执行命令。详见 [数据准备](docs/PUBLIC_DATA.md) 和 [基准协议](docs/PUBLIC_BENCHMARK.md)。

**独立公共协议无需旧的私有预处理文件或模型。** 它重新测量可复现的 CPU 基线；历史 CrossEncoder/LambdaMART 的 0.7748 不会被当作这个新模型的成绩。公开仓库包含源码、测试、协议与汇总证据；原始数据、图片、模型权重、索引、逐查询记录和本地个人数据由运行命令在本机生成，不入 Git。

## 研究流程与诊断

- 固定上游提交与源文件哈希，按查询 ID 和归一化文本隔离 train/val/test，抽样不使用相关性等级。
- 全量保留每个查询的人工候选；候选数量可变，不人为挑选正负样本。
- 仅用验证集比较固定的 0/5/15/30 轮训练，冻结后才执行测试；模型产物可直接被服务加载。
- 分别评估人工候选的 NDCG/Exact@1 与真实全库召回的已知正例 Recall@10/100，报告配对 bootstrap 区间、标注覆盖率和串行 p50/p95 延迟。
- 输出长查询、数字、品牌和否定词诊断切片与失败查询列表；测试结果只用于分析，不继续调参。
- 服务核对模型、目录和报告归属，展示相同召回池中的排序变化及线性特征贡献。贡献只是模型算术分解，不是因果解释。

## 不下载数据时的功能检查

```bash
python -S scripts/demo.py
python -S -m unittest discover -s tests -v
```

`demo.py` 使用 6 个手写虚构商品检查真实索引、品牌排除、数字匹配和缓存，零依赖、零网络。这仅是 smoke test；上面的真实数据入口才是完整实验。缺少可选依赖的研究模块测试会明确跳过。GitHub Actions 配置在 Python 3.10/3.12 下运行标准库测试。

本地公开副本验证：112 项测试，当前 CPU 环境 110 项通过、2 项因缺少 FAISS 跳过；标准库环境 83 项通过、29 项跳过，均无失败。另含真实 Parquet 导入契约测试与实际 HTTP 服务测试。详见 [验证记录](reports/public-cpu-v1/verification.json)。

## 实现了什么

### 新公共流程的实际结果

固定 100 条测试查询、3,741 个人工判断对；训练 98,551 个不同等级的商品对，验证选择 30 轮。

| 范围 | 基线 | 学习排序 |
| --- | ---: | ---: |
| 全部已标候选 NDCG@10 | 0.7137 | **0.7494** |
| 全部已标候选 Exact@1 | 65.00% | 71.00% |
| 全库已知正例 Recall@10 | 19.35% | 19.07% |
| 全库已知正例 Recall@100 | 57.25% | 57.25% |

候选内 NDCG 增量的 95% 配对区间为 [0.0130, 0.0596]；全库 Recall@10 增量区间为 [-0.0152, 0.0097]，**不能确认端到端提升**。工作台因此明确展示词法基线与学习排序对照，不把后者称作全面更优。含至少 5 个 token 的长查询切片出现全库重排回退，需要在新的验证实验中处理候选分布偏移，不能根据本次测试直接调参。

实测选择训练共 426 秒，最终评测 44 秒；模型和目录驻留内存后，串行检索加重排 p50/p95 为 176/350 ms，应用层结果缓存关闭。此为本机实验成本，不是生产吞吐承诺。[完整原始汇总](reports/public-cpu-v1/report.json) · [失败分析与 JD 边界](docs/RESEARCH_ASSESSMENT.md)。

### 模块与范围

| 方向 | 实现 | 当前边界 |
| --- | --- | --- |
| 查询理解 | 品牌、数字、否定约束；Query/商品联合编码 | 主要评估英文标题与品牌 |
| 精排 | 冻结 MiniLM CrossEncoder + 15 维 LambdaMART | 学习词法约束与语义分数的组合 |
| 检索与资源 | SQLite 词法召回、FAISS/HNSW、int8/mmap、LRU 与结果缓存 | 八万级商品实验，未验证千亿级容量 |
| 多模态 | TinyCLIP 图文向量与轻量融合 | 独立 Fashion200K 代理任务，不能等同购物相关性 |
| 个性化 | 本地收藏/屏蔽；独立 Beauty BPR 实验 | 无真实线上用户画像或业务收益结论 |
| 生成式检索 | 小型 GRU 商品 ID 生成对照 | 旧实验明显弱于 BM25，默认关闭 |

82,427 商品版本默认使用词法召回与新精排；该目录的完整向量重建尚未完成，不计入本轮准确率收益。旧 15,171 商品版本保留向量与生成式对照模块。

## 历史 CrossEncoder / LambdaMART 结果

新的 ESCI 留出集排除旧实验所有查询 ID 和归一化查询文本。200 个测试查询、7,646 个官方人工标注 Query/商品对；每个方法排序完全相同的全部已标候选。模型只用训练/验证集选择，随后冻结并打开测试集。

| 方法 | NDCG@10 | 前十 E/S 比例 | 首位 E 比例 |
| --- | ---: | ---: | ---: |
| BM25 | 0.7108 | 82.75% | 68.50% |
| 旧 pairwise 权重迁移 | 0.7530 | 85.35% | 73.00% |
| CrossEncoder int8 | 0.7392 | 84.20% | 72.50% |
| 验证集选定的新模型 | **0.7748** | **86.00%** | **80.50%** |

E 为精确匹配，S 为可替代品。NDCG@10 是排序质量指标。相对旧模型，新模型 NDCG@10 增加 0.0218，查询配对 bootstrap 95% 区间为 [0.0081, 0.0355]；首位 E 比例增加 7.5 个百分点，区间为 [2.5, 12.5] 个百分点。前十 E/S 比例的增量区间跨零。

另在预先选定的 50 个测试查询上，从完整目录召回相同 top100 候选后重排：新模型已知 E/S Recall@10 为 16.45%，词法顺序为 15.36%，独立 CrossEncoder 为 17.07%。新模型相对词法顺序的增量区间为 [-0.0035, 0.0259]，**目前不能确认稳定的端到端提升**。未标商品不能当作全部不相关，也不能用人工候选池的 86% 代替全库搜索准确率。

完整说明：[结果](docs/ACCURACY_RESULTS.md)、[实验协议](docs/ACCURACY_PROTOCOL.md)、[数据划分](docs/DATA_V2.md)、[机器可读汇总](reports/accuracy/README.md)。所有数字都是离线研究结果，不代表线上 CTR、转化率或中文搜索效果。

## 历史神经精排实验的准备

1. 阅读 [第三方来源与许可](docs/THIRD_PARTY.md)，自行取得所需数据与模型。ESCI 固定上游版本与原始文件 SHA256 见 [数据说明](docs/DATA_V2.md)。本仓库不代分发这些资源。
2. 使用独立 Python 环境。`requirements-accuracy.txt` 记录历史实验的实际推理环境；完整功能与训练依赖范围在 `pyproject.toml`。历史动态 int8 配置使用 oneDNN，版本与后端都属于冻结实验的一部分。
3. 准备数据、训练并冻结模型，完成最终评测后，才可使用 `run-accuracy.ps1` 启动完整网页服务。入口会检查缺失产物；默认端口为 18866。通过 `ECOM_SEARCH_PYTHON` 指定解释器，未设置时使用 PATH 中的 Python。

```bash
# 下载并验证固定版本 CrossEncoder 权重（需要网络和可选依赖）
python -m ecom_search.cross_encoder prepare --model outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2

# 先查看各步骤参数与输入要求
python scripts/prepare_data_v2.py --help
python scripts/train_accuracy.py --help
```

**历史成绩的精确复现限制：** 历史 v2 抽样排除了旧 400 个冻结查询。公开仓库不包含这些旧输入文件、原始标签、训练权重或逐查询输出；仅靠本仓库不能一键重现历史表格。精确复现需匹配协议记录的全部输入文件与环境哈希；使用不同输入时应建立新的 train/val/test 实验并重新报告结果，不要改写历史元数据冒充复现。前面的独立公共基准已经消除了对旧输入的依赖，但它是新的实验协议，应报告自己的结果。

`docs/` 中保留了完整离线交付环境的历史说明；其中 `data/`、`outputs/` 与模型路径表示需要另行生成的本地产物。公开发布只保留 `reports/` 中列明的汇总文件。

## 代码导航

- `ecom_search/`：检索、特征、重排、评估、服务与研究模块。
- `scripts/`：数据准备、训练、验证、离线核验和最小示例。
- `tests/`：功能、数值一致性、缓存边界与数据隔离检查。
- `web/`：本地研究服务界面。
- `docs/`：方法选择、数据与模型来源、实验结果和限制。
- `upstream/`：固定版本的 ESCI 上游参考与来源记录。

## 许可

项目代码按 [Apache-2.0](LICENSE) 发布，保留 [NOTICE](NOTICE) 与上游归属。第三方数据、图片及模型分别遵守其原始条款，项目代码许可证不扩展其使用权限。详见 [THIRD_PARTY.md](docs/THIRD_PARTY.md)。
