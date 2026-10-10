# 来源与本地交付

新增代码在 Amazon ESCI 的公开任务上独立实现。`upstream/PROVENANCE.json` 记录导入的官方 commit 与文件 SHA，`upstream/` 仅导入 README 和两份原始排序代码，并非完整 fork。官方 Apache-2.0 LICENSE/NOTICE 原样保留。

MiniLM 本地缓存来自 [Sentence Transformers 的官方模型页](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2)，该页声明 Apache-2.0。本项目固定并检查实际本地模型文件 SHA，复制相同内容到 `outputs/text-assets/minilm` 供迁移加载。当前工作区未追溯缓存的完整远端 commit，因此只声称内容指纹固定。许可证副本在 `docs/licenses/MiniLM-Apache-2.0.txt`。

v0.3 的 [MS MARCO MiniLM L6 CrossEncoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a) 固定 revision 为 `233902d25c440f23af6f7d6e94d2946bac0bee0a`，官方模型元数据声明 Apache-2.0。模型 README、下载来源、Git/LFS 校验值、所有实际文件 SHA 随 `outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2/model_manifest.json` 保留。原始权重在本机动态量化，未额外训练其参数。LightGBM 仅作为训练库使用，服务读取独立导出的数值树，不随项目复制 LightGBM 二进制。

Fashion200K 与 TinyCLIP 的固定版本、图片来源及权利声明见 `docs/MULTIMODAL.md`、`data/multimodal/manifest.json` 和 `data/multimodal/license_manifest.json`。完整原作者许可证文本在 `data/multimodal/licenses/`，交付保留对应模型 README/source manifest；第三方照片权利不因代码许可证而转让。

Luxury Beauty 数据来自 UCSD Amazon 2018 公开研究数据。原站未明确授予本项目新的开放再分发许可，具体来源与使用边界见 `docs/PERSONALIZATION.md` 与实验 manifest。完整本地交付只保留服务所需的训练期匿名评分记录、商品目录与冻结模型，不含原始评论正文、下载压缩包或逐测试用户轨迹；本地包不等于可以公开上传全部研究数据。

小包包含标准库文本服务、源码、ESCI 冻结样本与聚合结果；完整包额外包含图文数据、本地权重和研究服务产物。两包均排除本机收藏 SQLite、上传文件、Python 环境、缓存和测试临时目录。所有入包文件逐项 SHA256 与 ZIP CRC 校验。
