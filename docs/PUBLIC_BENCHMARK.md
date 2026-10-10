# 从官方数据到可服务模型：独立 CPU 基准

这条流程从官方 ESCI Parquet 建立自己的训练、验证、测试集，训练可解释的 pairwise logistic 排序器，并输出可被真实搜索服务直接加载的 SQLite 索引和模型。它不需要旧项目的 `outputs/ranking/model.json`、旧 400 条 query、神经网络权重或任何历史测试缓存。

`scripts/demo.py` 的手写数据只验证软件能运行。本文的默认流程使用真实人类标注，两者结果不能混用。历史 CE/LambdaMART 的 NDCG@10=0.7748 来自另一冻结实验；本流程不会把它当作新训练的成绩。

## 执行

在仓库根目录运行；需要 Python 3.10+，数据转换依赖 pyarrow，训练和评测只依赖标准库以及 Python 自带的 SQLite FTS5。

```bash
python -m pip install -r requirements-public.txt
python scripts/prepare_public_esci.py --download
python scripts/benchmark_public.py select --data data/esci-public --output outputs/public-cpu-v1
python scripts/benchmark_public.py final --data data/esci-public --output outputs/public-cpu-v1 --per-query
```

如已下载官方数据，使用 `--raw path/to/parquets`，省略 `--download`。导入器仍会验证固定版本和完整文件 SHA256；文件命名与更多数据说明见 [PUBLIC_DATA.md](PUBLIC_DATA.md)。默认 1,000/100/100 条 train/val/test query，保留每个 query 的全部官方已标注候选，最少 20 个。目录是这些候选商品的并集，**不是整个 Amazon 商品库**；测试 query 的商品文本可以被检索和建立无监督词频统计，但其 relevance 标签不会用于模型选择。

首次下载约 1.16 GB。索引、训练对和词法特征在 CPU 内存中工作；扩大查询数会提高商品数和内存需求。训练没有后台下载、GPU、外部服务或收费 API。

## 为什么分为两条命令

`select` 只 JSON 解析商品、train 和 val。它将 test 文件视为不透明字节计算 SHA256，不解析测试标签，不生成 test 指标。它在训练集上分别拟合固定 epochs `[0, 5, 15, 30]`，其余参数固定为 learning_rate=0.05、l2=0.001、seed=42；按验证集宏平均 NDCG@10 选择最高者，平局选择较少轮数。0 轮是诚实的未训练候选，允许验证集不支持训练改进的情况。

特征包括标题 BM25、token 覆盖、字符三元组、品牌、数字/单位、相邻词组、短语和否定线索，共 12 个。每个训练 query 最多抽样 100 个不同等级的人类判断对；不会把未标注商品当负例。中文、多义词消歧、单位换算和真实属性合规均没有被这些启发式证明。基准与线上 `SearchEngine` 使用同一 `plan_query(...).ranking_query` 规范化文本。

选择结束输出 `model.json`、`products.sqlite`、`selection.json`、`train_val_identities.json` 和 `freeze.json`。模型携带商品目录指纹，可直接由现有服务加载；冻结文件绑定数据、相关代码、索引、模型及选择记录。`final` 在读取测试标签前核对这些指纹；改变任何绑定文件会拒绝评分。它再检查 test 的 ID 和 NFKC/大小写/空白规范化 query 与 train/val 不重叠。

`final` 以独占方式写入 `final_started.json` 后才读取 test。已有结果、重复执行或中断后的直接复跑都会被拒绝，避免只保留满意结果。这是可审计的本地流程约束，不是防恶意修改的访问控制。若运行失败，保留该目录和失败记录；修复基础设施后应另开明确记录的运行，而不能把失败目录或已见过的测试集重新包装成从未使用的新 holdout。不要根据本次测试的差例继续调参并声称同一 test 仍未使用。

## 指标解释

`report.json` 将两种问题分开记录：

| 场景 | 对照 | 输出与边界 |
|---|---|---|
| 全部人工判断候选内排序 | 自定义标题 BM25 vs 验证选定线性排序器 | NDCG@10、P@10(E/S)、Exact@1、候选内正例 Recall@10 |
| 当前有限目录上的检索后排序 | SQLite FTS5 标题/品牌 BM25（字段权重 3:1）vs 同一 top100 候选上的线性重排 | 已知正例 Recall@10/100、至少召回一个已知正例的 query 比例、top10 判断覆盖率 |

NDCG 收益为 `2**grade-1`；E/S 为 grade>=2。零理想 DCG 或没有已知 E/S 正例的 query 只在对应指标的宏平均中排除，每个指标保留有效 query 数。短自定义候选集的 P@10 仍以 10 为分母。`judged_fraction@10` 以 10 为分母，是判断覆盖率，**不是准确率**。

商品库检索中，其他 query 对同一商品的标签不等于当前 query 的标签。未知结果保持未知，因此不输出全库 P@10/NDCG，也不能从这份稀疏 qrels 推导真实全库准确率。Recall@100 同时是这条召回链的已知正例候选覆盖率；重排只改变顺序，两种方法的 Recall@100 必然相同。

每个对照给出以 query 为单位的 2,000 次配对 bootstrap、差值和 95% 百分位区间。区间跨零时，不支持“稳定提升”的说法。短/长 query（>=5 token）、含数字、显式提及目录品牌的切片是诊断规则，没有人类复杂 query 标签，且未做多重比较修正。不能把某个正向切片选择性当作整个系统提升。

## 性能与差例

报告含串行单进程检索、重排、二者合计的 p50/p95 和平均耗时。模型与商品特征已加载，应用层结果缓存关闭，OS 缓存不受控；每条 query 在固定候选评测预热它的特征缓存之前先测服务链。加载特征耗时单列。这些数值不是并发吞吐、冷启动或生产 SLA。

`--per-query` 额外在本地生成 `per_query_test.jsonl`，包括 query、各通道结果 ID、指标、切片和延迟，用于追查召回漏失、品牌/数字错误及重排回退。它含原数据内容，被 `.gitignore` 排除，不应随公开源码提交。`report.json` 只含汇总及最差变化 query ID；`report.md` 是简表。服务索引和模型也留在本地。

完成训练后可直接查询或启动现有 API：

```bash
python -m ecom_search search "wireless headphones" --index outputs/public-cpu-v1/products.sqlite --model outputs/public-cpu-v1/model.json --pool 100
python scripts/serve_public.py --experiment outputs/public-cpu-v1 --port 18867
```

本文的新公共流程不承诺与以前机器上的历史实验 query 互斥，也不是对历史深度模型指标的复现。它提供的是可公开取得数据、从头训练、验证选择、冻结测试、服务复用的一条完整基线，后续深度/多模态改进应建立新的预注册实验再比较。

## 本次真实运行（2026-10-10）

默认公开导入命令准备了 43,222 个商品，train/val/test=1,000/100/100 个 query。固定网格的验证 NDCG@10 为 0 轮 0.711446、5 轮 0.735564、15 轮 0.735985、30 轮 0.736973，因此冻结 30 轮模型。使用 98,551 个不同等级训练对，整个选择阶段耗时 426.3 秒，SQLite 索引 14.97 MB。

唯一一次 final 对 100 条测试 query、3,741 个人类判断进行评分，未根据结果重训：

| 指标 | 基线 | 学习排序 | 学习减基线的 95% 配对区间 |
|---|---:|---:|---|
| 完全标注候选内 NDCG@10 | 0.713730 | 0.749428 | [+0.013043, +0.059581] |
| 完全标注候选内 P@10(E/S) | 0.819 | 0.835 | [-0.003, +0.036] |
| 完全标注候选内 Exact@1 | 0.65 | 0.71 | [-0.02, +0.15] |
| 目录检索已知正例 Recall@10 | 0.193530 | 0.190703 | [-0.015182, +0.009743] |
| 目录检索已知正例 Recall@100 | 0.572474 | 0.572474 | [0, 0]（候选集相同） |

这份结果支持固定候选内 NDCG 改进，**不支持端到端搜索相关性已经提高**。93% 的测试 query 在 top100 中找到至少一个已知正例。较长 query（38 条、>=5 token）的目录 Recall@10 差值为 -0.02637，探索性区间 [-0.04702, -0.00667]，需要单独调查；它还没有经过多重比较修正，不能作为调好同一测试集的理由。应对后续假设使用训练/验证和新的测试方案。

本机 Windows/Python 3.10.21 的特征加载耗时 18.99 秒；驻留后串行检索 p50/p95 为 18.68/99.34 ms，检索加重排为 176.09/350.34 ms，100 query 完整 final 耗时 44.10 秒。它说明线性模型的 CPU 特征提取也有成本；不能凭模型参数少就宣称低延迟。上述数值只代表本次机器与负载。完整机器可读证据由 `outputs/public-cpu-v1/report.json` 和 `selection.json` 生成，公开仓库的汇总副本不包含 query 原文或原始数据。
