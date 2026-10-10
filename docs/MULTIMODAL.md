# CPU 真实图文商品检索实验

这个模块提供独立的服饰图文目录、图片检索、文字检索、图文融合，以及只训练小投影层的 CPU 实验。400 张图片来自同一个真实数据源的 200 个商品，不与 ESCI ASIN 拼接。

实验评价的是**同一商品不同视图的配对检索代理任务**。查询图片与目标图片的 SHA256 不同；查询文字直接保留来源的短 `category3` 字段，目录文字保留来源的 `text` 字段。来源没有提供这些查询的人工搜索相关性标注，也未在 dataset card 解释长文字的标注方法，因此本项目不把它称为人工购物查询评价。

## 来源与体积

- 数据：[Fashion200K 原作者仓库](https://github.com/xthan/fashion-200k) 与 [Marqo 的固定版本](https://huggingface.co/datasets/Marqo/fashion200k/tree/b9d0e6c9c14337620e442c93f40f7906942fe893)。两处声明 Apache-2.0；HF 发布者声明并不拥有原数据。商品照片保留第三方权利，保存来源与许可声明，下载用于本地研究示范。
- 模型：[TinyCLIP 官方实现](https://github.com/microsoft/Cream/tree/main/TinyCLIP) 与 [作者发布的固定权重](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M/tree/a2a8c6eaa2549ad66eb7c31b85022bf58273a26c)，声明 MIT。模型总计约 23.4M 参数，safetensors 文件为 93,812,468 bytes，SHA256 为 `9339ee3d736344d0ddcaa6c03edc9f89688f08caaea5401220885233da726fcc`。

本次实际网络传输 119,751,951 bytes，包括模型。数据通过严格 HTTP Range 只读取第一个 Parquet shard 的前 11 个 row group，共 23,697,332 bytes；完整 shard 为 368,297,137 bytes，**没有下载或验证其完整 SHA256**。每个已读 range 和提取图片均记录自己的 SHA256。当前子集 200 个商品全部属于 `dresses`，不是整个服饰市场的代表样本。

`data/multimodal/manifest.json` 保留版本、来源、许可、网络字节数、数据文件哈希、图片哈希、Parquet range、抽样方法和 split 数量。抽样和 split 只依据商品 ID 的固定哈希，不依据模型结果。训练、验证、测试分别有 120/40/40 个不同商品；400 张图片的 SHA256 全部不同。

`data/multimodal/licenses/` 原样保留 Microsoft TinyCLIP 的 MIT 通知及 Fashion200K 原作者的 Apache-2.0 通知；`license_manifest.json` 记录固定 Git 提交 URL、完整内容 SHA256 和字节数，供完整本地交付保留来源与权利说明。

## 运行

默认主项目不需要多模态依赖。此模块需要 CPU PyTorch、Transformers、NumPy、Pillow；准备数据额外需要 PyArrow。请先创建项目虚拟环境并安装可选依赖，再从项目根目录运行。

```powershell
$mmPython = ".\.venv\Scripts\python.exe"
& $mmPython scripts/prepare_multimodal.py --output data/multimodal --assets-root outputs/multimodal-assets --products 200 --max-transfer-mb 190
& $mmPython -m ecom_search.multimodal benchmark --data data/multimodal --model outputs/multimodal-assets/tinyclip --output outputs/multimodal --threads 2 --batch-size 8
& $mmPython -m ecom_search.multimodal search --query "black floral dress" --mode text --k 5
& $mmPython -m unittest discover -s tests -p test_multimodal.py -v
```

准备脚本只下载固定公开资源，限制累计传输字节数，支持分段缓存，拒绝服务器忽略 Range 时的全量传输。准备完成后，模型推理和训练全部使用本地文件并开启离线模式，不自动下载模型。标准 `python` 环境安装这些可选依赖后也可运行相同入口。

图片与联合搜索使用真实 query 图片路径；可从 `data/multimodal/test.jsonl` 读取 `image_path`，然后加上数据目录前缀：

```powershell
$mmExample = Get-Content -LiteralPath data/multimodal/test.jsonl -TotalCount 1 | ConvertFrom-Json
$mmPicture = Join-Path "data/multimodal" $mmExample.image_path
& $mmPython -m ecom_search.multimodal search --image $mmPicture --mode image --k 5
& $mmPython -m ecom_search.multimodal search --query $mmExample.query --image $mmPicture --mode learned --k 5
```

## 融合与训练

图像与文本使用同一个 TinyCLIP 的归一化 512 维表示。TEXT 只比较文字，IMAGE 只比较图片，FUSION 对图文向量各取 0.5 后再次归一化。LEARNED 为两种表示分别学习一个对角投影，并学习文字权重 `gate`；该 gate 也用于目录侧，缺少某种查询输入时只投影并归一化已有输入。

编码器保持冻结。共训练 1,025 个参数：两个 512 维对角投影，加一个 gate logit。对角投影的权重经过裁剪后取指数，训练用对称 InfoNCE、温度 0.07、Adam 学习率 0.02、对角权重 L2 系数 0.01，CPU 2 线程，seed 为 20261009。训练负例只来自 TRAIN 商品，验证和测试商品不参与梯度计算。

优化步数固定为 `[0, 25, 75, 150]`。先在验证集最大化配对 Recall@1，再以 Recall@5 和较少步数打破平局；保存 `selection.json` 和 serving checkpoint 后才读取测试查询。此次选择 75 步，文字 gate 为 0.27666089，CPU 训练耗时 11.136 秒。

## 实测对照

200 个候选商品、40 个未参与选参的测试商品，exact cosine 搜索：

| 方法 | Pair Recall@1 | Pair Recall@5 | Pair Recall@10 |
| --- | ---: | ---: | ---: |
| TEXT | 7.5% | 20.0% | 35.0% |
| IMAGE | 60.0% | 77.5% | 82.5% |
| FUSION 0.5 | 40.0% | 72.5% | 75.0% |
| LEARNED projection + gate | 67.5% | 87.5% | 90.0% |

这些数字表示来源商品 ID 的配对检索命中率，不表示购物满意度。固定融合在本次数据上弱于纯图片，说明加入文字需要验证；学习后的融合优于固定融合和纯图片，但测试只有 40 个商品，不作统计显著性或线上收益声明。模型预训练与这些旧网页照片是否重叠也未知。

`outputs/multimodal/report.json` 包含各 split 的真实图像/文字编码耗时与 cache hit、训练 loss、完整 validation 对照、test 消融、检索耗时和限制。首跑 400 个图文对的 CPU 批量编码共约 119.61 秒；这个时间包括预处理，排除初始模型加载。检索 p50/p95 只测已缓存向量的余弦计算与排序，不包括编码器推理；图文编码耗时另列。缓存绑定模型与实际输入内容，读缓存时仍检查图片文件的 SHA256。

## 接入 API

```python
from ecom_search.multimodal import MultimodalIndex

index = MultimodalIndex("outputs/multimodal/selected")
text_results = index.search("black floral dress", k=10, mode="text")
image_results = index.search(image_path="/absolute/path/to/a/real.jpg", k=10, mode="image")
joint_results = index.search("black floral dress", image_path="/absolute/path/to/a/real.jpg", k=10, mode="learned")
reranked = index.rerank([item_id for item_id, _ in text_results], query="black floral dress", k=5)
product = index.catalog[joint_results[0][0]]
```

返回值为 `list[(id, cosine_score)]`；`index.catalog` 是 `id -> product` 的字典，包含 `title`、`text`、绝对 `image_path`、来源和图片哈希。`search_vectors()` 可用于已经编码的输入；`ClipEncoder.query()` 返回已有模态的向量字典。`k=0` 返回空结果，`k>N` 返回全部候选，负数、未知重排 ID、形状不一致和损坏的缓存/模型 artifact 都会拒绝。

Serving 目录保存 `catalog.json`、`gallery.npz`、学习参数 `model.json`、包含来源模型/选择/内容哈希的 `metadata.json`。默认加载会延迟初始化编码器，检查固定本地模型内容后再执行查询。

完整项目移到另一个目录后，若 checkpoint 中旧的绝对路径失效，会回退到模块所在项目的 `data/multimodal` 与 `outputs/multimodal-assets/tinyclip`。仍校验原 manifest、图片与模型内容 SHA256；只重新定位内存中的路径，不改权重或验证集选择。源码仓库不包含模型或图片时，需要先运行准备和 benchmark；完整本地 ZIP 可以保留这些已准备文件。
