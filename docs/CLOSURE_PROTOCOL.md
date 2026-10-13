# Closure v1：一次新查询检验与发布门槛

前次消融暴露了两个需要验证的问题：自然语言否定的硬过滤可能损失召回；人工候选池训练的排序器在真实召回池上未必改善前十结果。本次在**排除全部已登记历史查询**的新队列上固定六个配置，用验证集选一个，只对冻结赢家与基线进行一次最终测试。结果允许保留基线，不预设新方法一定提高准确率。

## 数据及范围

准备细节、原始文件版本、身份排除和容量调整见 [CLOSURE_DATA.md](CLOSURE_DATA.md)。固定为 800 train / 200 validation / 300 test，46,249 件人工判断商品的并集。新 query 的官方 ID 和归一化文本都与此前 3305 个历史 query 互斥。商品、语义改写、预训练模型语料不在这一排除保证内。

指标是有限商品库上的已知 E/S 正例 Recall，未判断 query-product 对保持未知。候选池的已知正例Recall@100衡量覆盖，最终Recall@10衡量排序后前十；judged@10表示前十结果中被人工判断的比例，不是Precision。无已知正例的query保留在总体中，其Recall未定义并单独计数。

## 固定的修正候选

| 维度 | 配置 |
|---|---|
| 查询约束 | `legacy`：沿用旧自然语言硬过滤；`soft`：取消自然语言否定的硬过滤，只有显式 `-word` 保持字面硬过滤 |
| 排序 | `bm25`：SQLite title+brand BM25；`judged`：原始人工候选训练；`retrieved`：soft top100与已有人工qrels交集训练 |

两维组成六个配置。两种学习模型都使用同一12维特征、pairwise logistic训练30轮、学习率0.05、L2=0.001、seed=20261013，每query最多100个偏好对。召回pool固定100。`retrieved` 不把未判断召回商品当负例；没有可比较等级的query不产生偏好对，其数量进入训练报告。修改训练池会改变样本数、难度和分布，不能把结果解释为单一特征的独立因果效应。

`soft_judged` 和 `soft_retrieved` 仍可使用学习排序中的否定特征；`soft_bm25` 只按BM25排序，不使用否定排序特征。因此“取消自然语言硬过滤”不能一概描述成“用软排序替代硬过滤”。

验证集先筛掉候选Recall@100低于基线均值的配置，再选Recall@10最高者。准确平分时按照固定顺序、基线优先。验证集六行全部保留；同验证集内的硬/软对照和训练池对照仅用于探索解释，不能相加贡献，也不当作独立测试证据。

## 冻结与最终门槛

顺序为 `prepare → train → select → final`。每阶段绑定源码、数据和上阶段产物SHA-256；训练与选择阶段仅哈希test文件字节，不解析标签。独占的阶段开始标记防止同目录重跑被悄悄覆盖。`select` 固定赢家与配置后，`final` 才读取test，且只评估基线与这个赢家。

基线为 `legacy_bm25`。最终单一主要比较为赢家减基线的宏平均已知正例Recall@10；按query配对bootstrap 2000次、seed=20261013得到95% percentile区间。发布同时要求：

1. Recall@10差值区间下界严格大于0。
2. 候选Recall@100差值区间下界至少−0.01（允许的退化上限为1个百分点）。
3. 验证集赢家并非基线本身。

候选覆盖是与主要指标一起满足的约束，未通过任一条件就不晋级；服务保持基线。置信区间只反映query抽样不确定性，不包含训练种子或部署域变化。此次没有根据最终test结果再调参、再选六个配置或重做测试。

离线与服务共用 `ecom_search/closure.py` 的查询规划、召回和排序。服务加载 `release.json` 后重新校验数据、源码、模型、索引、报告、选择结果与门槛；不能在HTTP请求里切换到未通过门槛的模型。线上展示为基线与实际发布配置，分别带各自召回池。

## 新克隆从零复现

Python 3.10及以上。CPU版本不需要GPU、模型下载或在线LLM；只有Parquet准备需要pyarrow。原始数据约1.16GB，索引、特征和训练另需本地空间和运行时间。

```bash
pip install -r requirements-public.txt
python scripts/prepare_closure_esci.py prepare --download
python scripts/benchmark_closure.py prepare
python scripts/benchmark_closure.py train
python scripts/benchmark_closure.py select
python scripts/benchmark_closure.py final
python scripts/serve_closure.py --port 18868
```

保留服务进程运行，在另一个终端回放全部封存test预测并导出报告：

```bash
python scripts/verify_closure_serving.py
python scripts/summarize_closure.py --output reports/closure-reproduced --doc docs/CLOSURE_REPRODUCED.md
```

HTTP回放比较每个query在基线和实际发布配置下的完整候选顺序、全部排序ID/分数与显示Top10；只读取封存预测作一致性验证，不重算准确率或改变发布决定。结果存在时，汇总器也会校验其release/report/model/UI指纹并导出。

已缓存官方Parquet时，第一条prepare改用 `--raw /path/to/official/esci`。数据默认写入 `data/esci-closure-v1`，实验默认写入 `outputs/closure-v1`；已有目录或阶段标记会报错，不能通过删除标记把重复试验伪装成首次运行。仓库自带的历史汇总目录已经存在，所以复现命令用新的报告目录，保留原结果。

服务地址为 `http://127.0.0.1:18868/`，`/health` 返回release状态和指纹。即使最后未晋级，也能启动基线服务并展示保留它的证据。

## 本次执行目录与中断记录

本地第一次 `outputs/closure-v1` 在特征缓存阶段为修正benchmark完整性检查而停止。它保留protocol、training-start标记和索引，没有完成训练、验证、选型或test。第二次在 `outputs/closure-v2` 使用相同数据、训练配方、六个配置、验证规则和发布门槛重新运行；这不是观察到test结果后重选方法。

| 协议 | SHA-256 |
|---|---|
| 第一次，中断 | `7cb9cd4f9710a04fdb2e816faeaff44302d3c8413209bc3096b056f94fc1c02f` |
| 第二次，修正完整性检查后 | `58c2a88310da21bc557787598959f0a0fa242d51ac0be7bb54c5437254d4e3da` |

两份协议只有注册时间与 `scripts/benchmark_closure.py` 的源码指纹不同。汇总器会检查首次目录中不存在validation/test结果，并导出两份协议和执行记录；不会遗漏已有试验结果。

本次汇总与服务的命令为：

```bash
python scripts/serve_closure.py --run outputs/closure-v2 --port 18868
# 在另一个终端：
python scripts/verify_closure_serving.py --run outputs/closure-v2
python scripts/summarize_closure.py --run outputs/closure-v2 --previous-run outputs/closure-v1
```

汇总器只读取聚合结果与身份/指纹元数据，验证绑定后导出公开证据；不评估query、不重新选择模型，不导出query原文、qrels、逐query预测、模型权重或索引。训练摘要仅保留偏好对数、覆盖、loss和成本。最终实际决定以 [CLOSURE_RESULTS.md](CLOSURE_RESULTS.md) 及 `reports/closure-v1/release.json` 为准。

该闭环仍是单机、小规模、有限人工判断商品库中的实验与服务一致性验证。它没有证明多模态个性化、千亿级检索、线上转化收益或分布式资源优化，不能据此声称已覆盖整个JD。
