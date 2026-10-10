# GitHub 项目筛选与本轮选择

2026-10-09 使用 GitHub 插件读取真实仓库 README、许可证和关键训练代码。选择依据为这份 TikTok Shop JD 的搜索相关性、数据真实性、普通电脑可运行性和可明确比较的改进点。

| 项目 | JD 相关性 | 选择理由或成本 |
|---|---|---|
| [Amazon Science ESCI](https://github.com/amazon-science/esci-data) | 电商 Query 与商品排序、语义匹配 | 主线。官方人工 E/S/C/I，Apache-2.0，能冻结 Query split 和独立标签；原始 products 文件约 1.1 GB，本轮复用已校验的轻量标题/品牌子集 |
| [FlagEmbedding / BGE-M3](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3) | 多语言、dense/sparse/multi-vector 检索 | 进阶对照。比 MiniLM 更大的模型增加 CPU 成本；多向量开销应单独量化，不在本轮默认运行 |
| [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) | Semantic ID 与生成式检索思路 | 原任务为历史序列推荐；要匹配商品搜索须新增 Query→Semantic ID 训练，默认嵌入与训练规模不适合普通电脑直接复现 |
| [CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir) | 图像 + 相对描述的组合检索 | 适合后续独立多模态实验。需 FashionIQ/CIRR 图像和标签；官方 README 的 Combiner 特征预计算缺口可做具体代码贡献 |

## 为什么先做 ESCI

完整项目的价值在于“可解释改动 + 独立标签 + 固定对照”。单纯启动大模型 API 很难说明算法改进。官方 [ranking/train.py](https://github.com/amazon-science/esci-data/blob/7916cdf6ab75a462e77f20ab40428a10923998d5/ranking/train.py) 使用 Query+title，以 E/S/C/I=1/.1/.01/0 的 gain 训练回归模型。本轮新模块提供 CPU 可训练 pairwise 目标、字段/属性特征、同 Query 监督和验证选择，保留上游原代码供审阅。

ESCI 没有用户画像、图片和价格。由此只能证明商品相关性排序与商品池检索，不证明个性化或多模态能力；增益和候选池不同也不能宣称超过官方模型。

## 下一步可做成独立实验的改进

1. 有 GPU 后，用同一 train/val/test 替换线性 ranker 为蒸馏双塔或小 CrossEncoder，比较字段消融、属性 hard negatives 和候选池排序。始终使用已有人工判断，未判商品不自动作负例。
2. 引入 FashionIQ/CIRR，扩展 CLIP4Cir 冻结编码器 feature cache。FashionIQ 随机使用两条 caption 的四种组合，缓存须覆盖全部组合，避免改变训练 augmentation；报告 Recall、epoch 时间、缓存体积和峰值显存。
3. 生成式搜索需要 Query-conditioned Semantic ID 学习与合法解码、碰撞审计；原 RQ-VAE 推荐结果不能直接当作搜索结果。比较 ANN 的资源和质量后再决定是否值得训练。

本轮已做的内容和未来实验分别记录，不把规划写成已实现成果。
