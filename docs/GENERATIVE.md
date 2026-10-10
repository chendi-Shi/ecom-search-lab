# CPU Query → 商品 Semantic ID 生成式检索

这个模块新增真实的 query-conditioned 生成模型，不把下一商品推荐当成搜索，也没有 nearest-neighbor fallback。冻结的本地 MiniLM 编码 Query；小型 GRU 自回归输出四层离散商品代码。每层八个 residual-kmeans code，码本仅在训练 Query 的 E/S 相关商品向量上拟合。验证、测试相关性标签不用于码本或索引构造。

全商品池向量通过固定码本得到 SID，构成无需相关性标签的 inference prefix trie。beam search 只扩展合法 prefix，累计模型本来的 token log probability。若 SID 对应多个商品，只在这个 collision bucket 内做 query-product cosine 比较；跨 bucket 保留生成概率的顺序。未生成 SID 中的商品不会凭向量相似度进入结果。

返回分数是保持上述顺序的 ordinal rank，不能解释为购买概率、相似度或校准后的相关性概率。软否定、数值约束等仍由原有搜索规划/训练排序模块负责。

在项目目录、已有 CPU torch/numpy/sentence-transformers 环境中运行：

```powershell
python scripts/train_generative.py --data data/esci --vectors outputs/ann/flat --bm25-index outputs/products.sqlite --output outputs/generative-v1
```

所有模型都从本地目录读取，不下载权重。已有输出目录含模型/实验产物时停止，避免覆盖已冻结实验；仅缓存目录可经 checksum 验证后复用。服务读取记录的本地 embedding 路径；迁移后该目录不存在时，回退到项目 `outputs/text-assets/minilm`，仍校验模型内容 SHA。构造器可传入 `encoder=existing_sentence_transformer` 共享冻结编码器。

固定训练方案是 CPU、两线程、seed=42、GRU hidden=128、token embedding=32、batch=128、AdamW lr=0.003、Exact 权重 1、Substitute 权重 0.5。预先指定 epochs `[30,60,100]` 与 beam `[10,30]`；只用验证 known-positive Recall@10 选择，平局选更少 epochs 和更小 beam。保存选定权重、码本、codes、ID 映射、embedding 模型和数据指纹后才打开最终测试标签。

产物 `outputs/generative-v1/report.json`/`report.md` 比较同一商品池的生成检索、SQLite BM25、相同 MiniLM 向量的 Flat cosine。评估 E/S known-positive Recall@10/100、无效 ID、SID 碰撞、展开候选量、额外 dot-product 数、CPU 时间和磁盘大小。检索耗时使用缓存 Query embedding，排除 MiniLM 编码；实际 API 的诊断单独保留 encoder 与端到端耗时。

API：

```python
from ecom_search.generative import GenerativeIndex

index = GenerativeIndex("outputs/generative-v1")
hits = index.search("black running shoes", k=100)  # [(product_id, ordinal_score)]
hits, diagnostics = index.search_with_diagnostics("black running shoes", k=100)
```

商品 vector 文件仍保存以解决 SID 碰撞，不能据此声称不需要索引存储。模型只有 250 个训练 Query；GRU 可能明显输给 BM25/Flat。离线结果没有工业规模、显著性或线上转化含义，未标注商品是未知而非负例。

本次真实 CPU 实验由验证集选择 60 epochs / beam=10。100 个最终测试 Query 的结果如下：

| 同 15,171 商品池模型 | Known-positive Recall@10 | Recall@100 |
|---|---:|---:|
| GRU 生成 SID | 0.019417 | 0.088862 |
| SQLite 标题+品牌 BM25 | 0.222674 | 0.668979 |
| 同 MiniLM 的 Flat cosine | 0.219722 | 0.725922 |

生成模型明显更差，因此作为研究模块保留，默认召回继续使用较强基线。这是实际训练结果，不是目标值。可训练解码器有 162,728 参数；训练加验证 414.55 秒；完整成功实验 504.12 秒。整个训练进程峰值 working set 为 712,593,408 bytes，包含冻结 encoder、框架和 Python；它不是单独 GRU 的内存。第一次未限制 BLAS 线程的尝试停在 cache/码本准备阶段，未做验证或测试，缓存已归档；成功实验显式限制 Torch/BLAS 两线程。

商品池产生 3,065 个 SID，94.67% 商品位于多商品 bucket，最大 bucket 为 115 商品。Test 约束解码无效 SID 为 0，但有效 ID 不等于相关商品。每 Query 平均展开 100 商品、做 98.83 次 bucket 内 dot product。权重文件 654,281 bytes，产物总计 24,706,128 bytes，其中完整碰撞向量占 23,302,784 bytes。详细 loss、验证选择、每 Query 指标和模型指纹见 `outputs/generative-v1`。

参考来源：生成式 token 检索思想来自 [TIGER / Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)，可检查的 MIT 开源复现是 [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)。本项目独立实现 query-conditioned GRU、train-only residual-kmeans、全 catalog 合法 prefix trie、bucket 内消歧和 val-only 实验协议；没有复制该仓库的历史推荐训练流程。这里是可验证的项目扩展，不宣称算法首创。
