# CPU CrossEncoder 重排

`LocalCrossEncoder` 提供冻结预训练模型的查询—商品联合编码分数。它读取真实文本，在 CPU 上输出单个原始相关性 logit；这些数值不是概率。模型、tokenizer、适配器源码、运行库版本、精度、量化引擎、输入长度、批大小、线程数和商品文本构造均进入配置指纹。适配器自身不训练，也不读取 qrels、划分文件或测试集。

使用 [cross-encoder/ms-marco-MiniLM-L6-v2 的固定官方版本](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a)，revision 为 `233902d25c440f23af6f7d6e94d2946bac0bee0a`。官方模型卡声明 Apache-2.0、英语及 MS MARCO passage-ranking 训练来源；这不构成电商效果证明。下载保存了模型卡、来源 URL、Git blob ID、所有文件 SHA256 与字节数。

模型权重 `model.safetensors` 为 90,870,598 bytes，SHA256 为 `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`。六个运行文件和 README 共 91,819,431 bytes，低于 100 MiB 下载预算。只下载 safetensors，不下载其他格式或执行远程代码。模型资产指纹为 `e4a71f3a566f01a667424c608ed6e25a207d4440d52799431f6144f9fd9ec7bb`。

显式准备模型时允许访问固定 revision 的官方文件；普通推理始终设置 `local_files_only=True`、`trust_remote_code=False`，冻结全部参数，使用 CPU 和 inference mode。默认 `precision="float32"`；可选 `precision="dynamic_int8"` 使用 `torch.ao.quantization.quantize_dynamic` 对 Linear 层动态量化为 qint8，固定 `onednn` 引擎，embedding 等其他部分保持 float32。缺少该引擎时直接拒绝该精度，不会回退 FP32。基础搜索项目无需导入 torch 或 transformers；只有调用可选重排时才加载它们。

```powershell
python -m ecom_search.cross_encoder prepare --model outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2
python -m ecom_search.cross_encoder score --query "running shoes" --document "Running shoes. Brand: Example" --document "Coffee mug. Brand: Example"
```

API 的默认配置为 `threads=2, max_length=128, batch_size=32`，tokenizer 使用 `longest_first` 截断。`product_text` 明确使用 `title. Brand: brand`，不加入 description，避免训练和服务阶段的文本口径发生变化。

```python
from ecom_search.cross_encoder import LocalCrossEncoder, product_text

ce = LocalCrossEncoder(
    "outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2",
    cache_dir="outputs/cross-encoder-cache",
    threads=2, max_length=128, batch_size=32,
)
scores = ce.score_pairs([(query, product_text(product)) for product in candidates])
# predict(pairs) is an alias; outputs retain the exact input order.
ranked = ce.rerank(query, candidates, k=10)  # [(product_id, raw_score), ...]
metadata = {"fingerprint": ce.fingerprint, "config": ce.config,
            "model": ce.model_info, "timing": ce.last_report}
ce.close()  # Drop local encoder references; cached scores remain reusable.
```

启用独立 int8 配置时，将构造函数增加 `precision="dynamic_int8"`，CLI 增加 `--precision dynamic_int8`。必须重新生成对应训练/验证分数与树模型，并冻结该配置；不使用 FP32 分数缓存或声称与 FP32 数值等价。

候选列表由调用方产生。CrossEncoder 只对传入候选重排，不会召回缺失商品。`rerank` 拒绝重复或空商品 ID；相同分数按 ID 排序。`score_pairs` 接受非空字符串对，保留重复输入的输出位置。FP32 同一次调用中只预测一次完全相同的文本对；int8 保留每个重复行及原顺序。非有限数值或输出数量不匹配会拒绝写入缓存。

缓存位于配置指纹前 24 位的子目录中，以适配 Windows 路径长度；metadata 和 SQLite 内仍核验完整指纹。FP32 的每行缓存绑定 exact query/document 内容 SHA256、完整配置指纹、有限 logit 及行校验和。动态 int8 的激活 scale 依赖 batch 上下文，因此只用独立 `complete_calls` 表缓存完整有序 `score_pairs` 调用：call key 包含全部 query/document 字符串、顺序、重复行和完整配置指纹，score list 使用完整 SHA256、shape、finite 校验后原子事务提交。改变同批次的其他商品、候选顺序、调用大小或 batch_size 会 cache miss，第一项也重新计算；不复用来自其他调用的单 pair 分数。

改 query、商品文本、tokenizer、权重或推理设置都会重新计算或拒绝加载，不会复用旧模型分数。额外的未登记 tokenizer/config/weight 文件也会拒绝加载。缓存元数据含模型绝对路径用于追溯，路径本身不参与数值指纹，搬迁后可将新模型目录传给构造函数。

实测 `outputs/cross-encoder-smoke/report.json` 使用旧 `data/esci/val.jsonl` 前五个 fixed20 query 的 100 个真实文本对，未使用标签，未访问 `esci-v2`、holdout 或 test。CPU 两线程、batch 32、max length 128 的 100 个不同文本对预测耗时 8.6047 秒，约 11.62 pairs/s；同样 100 个缓存命中耗时 0.0712 秒且分数完全相同。首次依赖导入和模型加载约 132.10 秒，本机工作集 485,474,304 bytes、峰值工作集 519,229,440 bytes、private commit 975,372,288 bytes。吞吐是本机这批短商品文本的结果，不能当作生产延迟。

这个早期 smoke 的配置记录先于缓存路径缩短及适配器源码指纹字段加入，报告保留原操作指纹 `aa3ab8631cce8fb34f0700b64e63a8bd8c53dc31ee0ded40f4b795c661f512a8` 以便审计；模型和数值规则相同。最终实验应从当前实例读取 `ce.fingerprint` 与 `ce.config`，由主实验协议冻结后统一评估。这里不宣称 ESCI 排序准确率改善，不使用留出集选择模型、融合权重或参数。

运行不下载模型的契约测试：

```powershell
python -m unittest tests.test_cross_encoder -v
```

11 项测试使用明确标记的假文件和注入预测函数，验证内容缓存、配置隔离、持久化复用、模型/tokenizer 篡改、缓存行篡改、非有限输出、重复 ID 和边界；int8 另外覆盖邻居改变使第一项重算、顺序/重复保留、完整调用命中、完整 SHA/shape/finite 检查与失败不提交。实际 FP32 模型另由上述 CPU smoke 验证；独立 `outputs/cross-encoder-smoke/benchmark_quantization.py` 在同样旧验证集 100 pairs 上实跑了 int8。修正为本机实际支持的 oneDNN 引擎后，正式适配器又用 40 个旧验证候选完成真实模型推理与重复调用缓存一致性检查，结果为 `outputs/cross-encoder-smoke/int8-real-report.json`。

独立量化报告 `outputs/cross-encoder-smoke/quantization_report.json` 的三轮中位耗时为 FP32 17.2015 秒、int8 4.8975 秒，约 3.512 倍加速。每 query Spearman 平均 0.9603，但 Top1 仅 3/5 一致，Top10 集合重合平均 0.92；raw logit 最大绝对差 1.0325。因此加速证据不能推出排序或准确率无损。测量包含共享 CPU 和文件 IO 影响，本轮将 int8 作为独立配置在训练/验证阶段重新比较，留出集不会用于决定精度。
