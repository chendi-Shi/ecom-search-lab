# 个性化偏好模块与真实电商离线训练

本模块提供两项独立能力：用户在本机明确收藏、隐藏或浏览商品后的个人偏好排序；真实 Amazon Luxury Beauty 用户评分序列上的 CPU BPR 训练。ESCI 只有 Query—商品相关性，本模块没有从它伪造用户、点击或购买历史，也没有把不重叠的 Beauty ASIN 拼接到 ESCI 商品上。

## 本机偏好接口

```python
store = PersonalizationStore("outputs/profiles.sqlite")
store.record("local-user", "us:B000EXAMPLE", "like")  # 真实用户明确输入
ranker = ProfileRanker(store, catalog)
adjustments = ranker.score(query, candidate_products, "local-user")
hidden_ids = ranker.hidden("local-user")
```

`record(user_id,product_id,action,timestamp=None)` 仅接受 `like/hide/history`，时间为 Unix 秒。`events(user_id,before_timestamp)` 严格读取截止时间之前的事件；`profile(user_id)` 返回收藏及隐藏 IDs；`clear(user_id)` 删除该本机用户资料。没有账号鉴权或云同步，用户标签只用于本机资料隔离，不是安全身份边界。HTTP 调用方应验证商品属于当前目录；不要自动把搜索曝光记录为喜欢。

`ProfileRanker.score(query,products,user_id,before_timestamp=None)` 返回 `{product_id: adjustment}`；无资料时为 0。显式隐藏由 `hidden()` 提供并由上层过滤。默认使用透明的词项哈希向量、30 日半衰期和事件数量置信度；这是未训练的本机偏好基线。传入实际缓存商品 embedding 字典可替换编码器。此处默认权重属于固定启发式，与下述 BPR 学习权重独立；不能把演示收藏算作真实用户电商效果验证。

## 真实电商评分训练

数据来自 [UCSD 官方 Amazon Reviews 2018 页面](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html)，选择其 **Luxury Beauty 5-core，34,278 条原始 review**。使用 [官方小 review 文件](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Luxury_Beauty_5.json.gz) 与 [官方 metadata 文件](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Luxury_Beauty.json.gz)。下载预算固定不超过 200MB，仅标准库即可解析与训练；不会读取评论正文、reviewerName 或图像。匿名原 user ID 再做 SHA256 哈希后存本机。

截至本次核查，来源页没有明确给出可重新授权的数据开放许可；代码许可不自动覆盖该数据。仓库交付代码和聚合指标，原始文件与用户逐事件产物保持本机，不宣称重新授权。按发布者要求引用 Ni、Li、McAuley 的 EMNLP 2019 论文 *Justifying recommendations using distantly-labeled reviews and fined-grained aspects*。官方下载路径无 immutable revision 或公开校验值：首次下载记录观察到的 SHA256，此后复跑检查字节一致；不将本地指纹声称为发布者签名。

采用 [BPR 原论文](https://arxiv.org/abs/1205.2618) 的用户—商品成对排序思路，独立实现 16 维用户与商品矩阵、固定 seed 和 logistic SGD。rating≥4 为已知正反馈；其余未点评商品仅作为均匀抽样的未判断比较项，不称作真实负反馈。评分时间是 review 发布日期，不是点击、曝光或下单时间。

## 时间与评测协议

1. 在实际事件时间上选择全局 80%/90% 时间分位点，所有相同时间戳留在同一切分；不按用户随机切分。训练只访问训练切分，商品池只包含训练期已出现的 ASIN。
2. 在验证集预先固定 epochs `[0,5,15,30]`、个性化 rank 融合权重 `[0,.25,.5,1]`。以已知正例 NDCG@10 选择；并列选更少训练轮次，再选更低权重。模型、选择理由、源指纹和时间边界在测试评分前保存。
3. 每个目标为未在更早历史中点评过、rating≥4 的实际商品；候选为该目标加至多 99 个固定 seed 均匀抽取的训练商品，排除**严格更早**的已点评商品。同时间戳及未来反馈不进入历史，也不据未来标签删候选。统一对照热门非个性化排序与已冻结 BPR 个性化排序。新用户回退热门，并单独报告冷/热用户切片。
4. 最多 1,000 个验证/测试事件按固定 SHA256 次序选取。指标为该抽样集合上的 known-positive Recall/NDCG@10；未抽到、未点评商品并无完整判断，不能冒充全库 Recall、CTR 或因果线上收益。

原始 5-core 的全时段用户/商品筛选本身利用完整资料，因此这不是前瞻人口样本。metadata 标题、品牌仅供显示，不进入训练，避免将抓取时的平均评分、销量或价格用作过去事件特征。单次 review 目标不提供购物 Query，因此本实验验证**电商个性化推荐**；它不能证明电商个性化搜索相关性已提升。

## 运行与产物

```powershell
python scripts/train_personalization.py
python -m unittest discover -s tests -p test_personalization.py -v
```

默认输出 `outputs/personalization/experiment`；已存在模型或报告时拒绝覆盖，复跑请指定新 `--output`。`report.json` 保存真实统计、时间边界、验证选择、冷启动切片和最终对照；`model.json` 保存测试前冻结的 BPR 权重；`per_event_test.jsonl` 保存逐事件排名和历史最大时间；`manifest.json` 保留来源及各切分文件 SHA256。原始数据保存在 `outputs/personalization/raw`，不用 GPU 或收费模型。

## 本次实跑结果（2026-10-09）

两份官方下载共 **14,989,293 字节**。34,278 条原始 review 按同用户、ASIN、时间戳去重，保留固定源文件的第一条，移除 5,219 条重复记录，得到 **29,059 条真实评分事件、3,819 用户、1,581 商品**。train/val/test 为 23,224/2,904/2,931 事件；train 截止为 2017-04-07 00:00 UTC，val 截止为 2017-08-27 00:00 UTC。训练已知商品池为 1,492 个。

验证选择为 **30 epochs、个性化融合权重 0.25**。最终固定 1,000 个测试事件、600 个不同用户：

| 排序 | Known-positive Recall@10 | Known-positive NDCG@10 |
|---|---:|---:|
| 非个性化热门 | 0.2630 | 0.154594 |
| 冻结 BPR 个性化融合 | 0.2540 | 0.159668 |

NDCG 差值 +0.005074；按用户簇重采样 2,000 次的 95% 区间为 **[-0.007311, 0.016291]**。Recall 差值 -0.0090，区间 **[-0.027888, 0.009009]**。两个区间都跨 0，结果没有证明显著收益；也不能因为 NDCG 略高忽略 Recall 较低。288 个测试事件来自训练未知用户，使用完全相同的热门回退。热用户事件 NDCG 0.094627→0.101754，仍须按同一不确定性与候选范围解释。

整个下载、解析、固定验证网格训练与最终测试约 **229.58 秒**，无需 GPU。模型、报告和来源保持本机；随后 bootstrap 仅分析已经冻结的逐事件预测，不重训或挑参数。

`BeautyRecommender("outputs/personalization/experiment")` 可加载实际冻结模型；`.users` 为可演示的训练用户标签，`.catalog` 为独立 Beauty 目录，`.recommend(user_id,k=5)` 返回带真实 title/ASIN 的推荐。加载先核对 model/catalog/train 文件 SHA256、模型的目录绑定和训练时间边界。服务只过滤训练期已点评商品，**不读取 val/test 事件来筛候选**；因此这是训练快照推荐演示，而非当前实时用户画像。它与本机 ESCI 收藏偏好搜索属于两个独立数据来源。
