# Closure v1：全历史排除后的新查询队列

此数据准备流程为消融后的修正提供一次新的查询级检验。它不会覆盖 `esci-public`、历史 `esci` / `esci-v2` 或旧实验结果，也不改变旧 importer。

## 在读取新标签前确定的协议

- 官方源固定到 `amazon-science/esci-data` 的 `7916cdf6ab75a462e77f20ab40428a10923998d5`，逐个验证两份完整 Parquet 的字节数与 SHA-256。
- 范围为 `product_locale=us`、`small_version=1`。每个 query 至少有 20 个不同的原始人工判断商品；不按相关性等级筛 query 或 candidate。
- 从官方 train 分出 **800 train / 200 validation**，从官方 test 取 **300 test**，种子 `20261013`。
- 原拟 1000 train / 200 validation；身份盘点发现：4003 个满足候选数门槛的官方 train query 中，2923 个已经进入历史实验，最多只剩 1080 个。故在新标签访问前，将 train 降为 800；所有历史排除、候选数门槛、validation 和 test 规模不变。此变动由身份容量决定。
- 排除历史 `esci-public`、`esci`、`esci-v2` 的全部 train / validation / test query，覆盖各目录内 corpus 与其他已登记 query JSONL。
- 同时按**官方 query ID**与 **NFKC + casefold + 空白归一化后的 query 文本**排除。其余重复文本只保留一个固定代表；跨官方 train / test 的重复文本排除。
- SHA-256 确定抽样、记录和 candidate 的顺序。`selection.json` 写入磁盘之后，才打开 `esci_label` 列复制完整 qrels。

新 test 的复制不等于评估；prepare 不计算标签分布、指标或任何模型分数。训练流程应单独读取 train 与 validation，不应调用会一并读入 test 的 `load_dataset`。

本次实际准备得到 **46,249 件商品**，train / validation / test 分别为 **30,667 / 7,699 / 11,306 对人工判断**。历史注册表覆盖 18 份身份文件、3305 个唯一 query ID 和 3305 个归一化 query 文本；新队列与它们两种身份的交集均为零。排除后符合候选数门槛的官方 train / test 分别还剩 1080 / 1425 个 query。这里的判断对数量来自候选身份计数，并未按等级统计。

可复现指纹：

| 对象 | SHA-256 |
|---|---|
| 排除注册表文件 | `fc99aadf8939a199630e79b8df800aad7b4ed962d34cbb8efc5403fce1744569` |
| 排除注册表规范内容 | `540dadcc097fb9a171f1077c2be2d11d66ac5c47dbd5d52caa57a40b6fc7b5e4` |
| 新 selection 规范内容 | `5a3f861f5dffddf93c0eca3a481d49002f1250532b5349ee28b775532de38b48` |
| 新 manifest 规范内容 | `18340c142414369b3ed13f8d99d2273db81e95cc3d231cb736d18f447804b24c` |

“规范内容”哈希使用排序键、UTF-8、无额外空白的 JSON，计算时排除对象自己的 `content_sha256` 或 `manifest_content_sha256` 字段；它与包含缩进和自身哈希的文件字节哈希不同。

## 可公开的排除注册表

[`benchmarks/closure-v1/exclusions.json`](../benchmarks/closure-v1/exclusions.json) 包含历史 query ID 与归一化文本的两个 SHA-256 集合。ID 哈希的输入为 `esci-us-` 加不带前导零的十进制官方 ID；文本哈希的输入为归一化文本的 UTF-8 字节。注册表还保存来源标签、文件名、字节数和 SHA-256，不包含原始 query、相关性标签或本地绝对路径。

所有历史身份文件须先与其原 manifest 的指纹一致，才开始提取身份。任何没有 manifest 指纹的额外 query JSONL 会报错，避免悄悄漏掉历史查询。注册表创建一次即冻结，已有输出不可覆盖；它的内容哈希和文件哈希都会进入新数据的选择记录和 manifest。

哈希用于重复检查和可复现性，不应被理解为加密匿名化：已知 query 文本可以计算同样的哈希。这里的源数据本身是公开 ESCI。

普通克隆者只需使用仓库内的注册表，无需持有我们的历史实验目录：

```bash
pip install -r requirements-public.txt
python scripts/prepare_closure_esci.py prepare --download
```

已有固定版本的官方 Parquet 时：

```bash
python scripts/prepare_closure_esci.py prepare --raw /path/to/official/esci
```

输出默认 `data/esci-closure-v1`，会拒绝覆盖已有目录。生成相同队列无需更改 seed 或样本数。`--download` 只下载缺少的文件，已有文件指纹不符会报错。原始数据和新 qrels 不提交到仓库。

维护者创建注册表的命令（仅在尚未存在注册表时使用）：

```bash
python scripts/prepare_closure_esci.py build-registry \
  --source public_v1=data/esci-public \
  --source historical_v1=../ecom-search-lab/data/esci \
  --source historical_v2=../ecom-search-lab/data/esci-v2
```

## 能证明与不能证明的范围

此流程证明新 query 的 ID 与归一化文本不出现在**登记的历史身份文件**中。它不保证语义改写完全互斥，也不保证商品互斥；预训练编码器的原始训练语料是否包含 ESCI 并未核实。候选商品库是抽中 query 的人工判断商品并集，仍然不是 Amazon 全量商品库。

封闭候选池保留完整判断，可计算 NDCG；跨 query 的全库检索存在未标注 query-product 对，Recall 只能按已知正例计算。数据准备只提取官方 title / brand，不包含图像、视频或用户行为。

`tests/test_closure_data.py` 覆盖 ID/Unicode 文本双排除、标签访问禁令、输入全量先校验、额外 query 文件、哈希篡改、拒绝覆盖以及真实 Parquet 的字节级确定性回放。
