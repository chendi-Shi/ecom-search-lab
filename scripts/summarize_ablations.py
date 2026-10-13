"""Render Chinese tables directly from the three published aggregate reports."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

GROUPS = {"title_bm25": "标题 BM25", "coverage": "token覆盖", "char_trigram": "字符trigram",
          "brand": "品牌匹配/冲突", "number": "数字匹配/冲突", "phrase": "bigram/短语", "negation": "否定约束特征"}
NEURAL = {"add_ce_to_lexical": "词法12维 → 加原始CE分数", "drop_bm25": "标题BM25",
          "drop_coverage": "token覆盖", "drop_char_trigram": "字符trigram", "drop_brand": "品牌",
          "drop_number": "数字", "drop_phrase_bigram": "bigram/短语", "drop_negation": "否定约束特征",
          "drop_ce_all": "全部CE信息", "drop_pool_stats": "CE池内统计合计",
          "drop_ce_raw": "原始CE分数（仍保留派生统计）", "drop_percentile": "CE池内百分位", "drop_zscore": "CE池内z-score"}
RETRIEVAL = {"full_k100": "完整基线 K=100", "no_brand_index_k100": "去品牌索引 K=100",
             "no_exclusion_filter_k100": "去硬排除过滤 K=100", "raw_tokens_k100": "原始token K=100",
             "full_k50": "完整流程 K=50", "full_k200": "完整流程 K=200"}


def ci(values, scale=1):
    return f"[{values[0]*scale:+.4f}, {values[1]*scale:+.4f}]"


def table(lines, headers, rows):
    lines += ["", "| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def render(reports, output):
    names = ("features", "neural", "retrieval")
    paths = {name: reports / f"ablation-{name}-v1" / "report.json" for name in names}
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    f, n, r = (loaded[name] for name in names)
    if not f["reference_reproduction"]["passed"] or not r["baseline_reproduction"]["exact_historical_rankings_checked"]:
        raise ValueError("Full reference reproduction is required")
    if any(value > 1e-12 for value in n["verification"].values()):
        raise ValueError("Neural reference reproduction failed")
    for name in names:
        if hashlib.sha256(paths[name].with_name("protocol.json").read_bytes()).hexdigest() != loaded[name]["protocol_sha256"]:
            raise ValueError(f"Changed protocol: {name}")
    lines = ["# 每个模块带来了什么：真实数据消融", "",
             "本次是已报告测试集上的**回顾性诊断**，用于解释已有结果。完整模型先重训复现，随后逐组删除并重新训练；另列冻结模型输入置零的敏感性实验。所有正面、负面和区间跨零结果均保留，没有根据这些结果选择新模型或更换默认配置。", "",
             "![三组消融的效果与置信区间](../reports/ablation-summary-v1/ablation.svg)", "",
             "图中A/B为NDCG差值×100，C为召回率差值（百分点）。各面板的数据、区间校正和比较方向分别标注，不能把效果相加。", "",
             "## 数据与比较范围"]
    table(lines, ["实验", "商品数", "train / val / test查询", "测试判断对"], [
        ["公共CPU特征消融", f["catalog_products"], f"1000 / 100 / {f['test_queries']}", f["test_pairs"]],
        ["历史神经精排", n["data"]["catalog_products"], " / ".join(str(n["data"]["queries"][s]) for s in ("train", "val", "test")), n["data"]["candidate_pairs"]["test"]],
        ["公共检索消融", r["catalog_products"], f"沿用原模型 / {r['test_queries']} test", "同公共数据"]])
    lines += ["两套数据的绝对分数不能直接比较。候选内NDCG使用完全相同且全部已标注的候选；全库实验仅统计已知E/S正例召回，未标注商品不当作负例。公共目录是有限的已标商品集合，不是完整Amazon库。", "",
              "所有去组效果依赖其余组件、特征之间的替代关系和训练配方，**不能相加为各模块贡献百分比**。固定配方消融也不等于为每个变体寻找了最优参数。", "",
              "## 1. 公共词法排序：七组特征分别重训", "",
              f"完整12维模型NDCG@10为 **{f['metrics']['fixed_pool']['full']['ndcg@10']['mean']:.6f}**。所有变体保持相同的1000个训练查询、30轮、学习率、初始化和采样种子。下表Δ=完整−去组，正数表示移除后变差。冻结敏感性是不重训直接置零，与去组重训回答不同问题。"]
    comparisons = f["comparisons_full_minus_ablated"]
    rows = []
    for group, label in GROUPS.items():
        delta = comparisons["retrained"][group]["fixed_pool"]["ndcg@10"]
        rows.append([label, f"{f['metrics']['fixed_pool']['without_'+group]['ndcg@10']['mean']:.6f}",
                     f"{delta['delta']:+.6f}", ci(delta["ci95"]),
                     f"{comparisons['inference_sensitivity'][group]['fixed_pool']['ndcg@10']['delta']:+.6f}",
                     f["activity"]["fixed_pool"]["groups"][group]["varying_queries"]])
    table(lines, ["移除组", "去组重训NDCG", "重训Δ", "95%配对区间", "冻结敏感性Δ", "候选间值变化的查询数"], rows)
    lines += ["这些95%区间未做多重比较校正，仅用于探索；跨零表示证据不足。描述字段全部为空，无法据此判断真实描述信息是否有价值。短语组是排序中的bigram/匹配，并非查询规划器的引号语法。删除否定特征列仍保留共享查询解析和检索过滤。", "",
              "同一个真实召回池上的重训结果如下。率的变化用**百分点**表示，区别于NDCG绝对差值。"]
    rows = []
    for group, label in GROUPS.items():
        delta = comparisons["retrained"][group]["corpus"]["known_positive_recall@10"]
        value = f["metrics"]["corpus"]["without_"+group]["known_positive_recall@10"]["mean"]
        rows.append([label, f"{value:.2%}", f"{100*delta['delta']:+.4f}", ci(delta["ci95"],100)])
    table(lines, ["移除组", "重训后已知正例Recall@10", "完整−移除（百分点）", "95%区间（百分点）"], rows)
    lines += ["所有排序条件共享每个查询的一次召回结果，因此Recall@100保持不变。人工候选池中的排序收益不能替代真实全库收益。[完整特征汇总](../reports/ablation-features-v1/report.json)。", "",
              "## 2. 神经精排：CE语义、词法约束与池内统计"]
    methods = (("bm25", "BM25"), ("cross_encoder", "单独CE"), ("without_ce_all", "词法12维LambdaMART"),
               ("without_pool_stats", "词法12维+原始CE"), ("full15", "完整15维：再加池内统计"))
    table(lines, ["方法", "NDCG@10", "Exact@1"], [[label, f"{n['methods'][name]['test']['ndcg10']:.6f}",
          f"{n['methods'][name]['test']['exact_at1']:.2%}"] for name, label in methods])
    lines += ["第一行对比是加入CE；其余都是完整15维−去组重训。保持同一7叶、200轮LambdaMART配方，不重新调参。NDCG区间对13个预设比较作Bonferroni校正，单比较置信度99.6154%，bootstrap覆盖率为近似。"]
    rows = []
    for name, label in NEURAL.items():
        delta = n["contrasts"][name]["metrics"]["ndcg10"]
        low, high = delta["familywise_ci"]
        conclusion = "正向区间未跨零" if low > 0 else "负向区间未跨零" if high < 0 else "未确认独立收益"
        rows.append([label, f"{delta['delta']:+.6f}", ci([low, high]), conclusion])
    table(lines, ["比较 / 移除组", "NDCGΔ", "多重比较校正区间", "本次证据"], rows)
    ce = n["contrasts"]["add_ce_to_lexical"]["metrics"]["ndcg10"]
    stats = n["contrasts"]["drop_pool_stats"]["metrics"]["ndcg10"]
    neg = n["contrasts"]["drop_negation"]["metrics"]["ndcg10"]
    lines += [f"加入原始CE的差值为{ce['delta']:+.6f}，池内统计合计为{stats['delta']:+.6f}，完整模型相对去否定组为{neg['delta']:+.6f}；强弱结论按上表校正区间判断。不能因某个去组模型点估计较高就替换默认模型。", "",
              f"否定组在{n['feature_activity']['test']['negation']['varying_queries']}/{n['data']['queries']['test']}个查询中有候选间变化。去组重训可能改变其他查询的排序，整体差值不是这些活跃查询的直接因果贡献。去原始CE分数仍保留两项CE派生统计，只有去除全部三列才是真正的无CE对照。", "",
              f"完整模型val/test最大预测误差分别为{n['verification']['val_max_abs_prediction_error']:.1g}/{n['verification']['test_max_abs_prediction_error']:.1g}，通过历史模型复现。实验复用经过指纹验证的历史特征缓存，重训耗时不包含重新执行CE推理。[全部神经结果](../reports/ablation-neural-v1/report.md)。", "",
              "## 3. 全库检索：候选覆盖与最终Top10分开", "",
              "所有配置沿用同一冻结排序器。候选池召回使用全部返回条目；Recall@100最多使用前100条，K=50时只有50条。移除品牌索引会重建字段TF/DF和长度统计，不只是把一个权重置零；重排仍使用原始品牌信息。"]
    rows = []
    for name, label in RETRIEVAL.items():
        row = r["methods"][name]
        rows.append([label, f"{row['candidate_count']['mean']:.2f}", f"{row['retrieval']['candidate_known_positive_recall']['mean']:.2%}",
                     f"{row['retrieval']['known_positive_recall@10']['mean']:.2%}", f"{row['reranked']['known_positive_recall@10']['mean']:.2%}",
                     f"{row['reranked']['known_positive_recall@100']['mean']:.2%}"])
    table(lines, ["配置", "平均候选数", "候选池召回", "词法Recall@10", "重排Recall@10", "重排Recall@100"], rows)
    lines += ["下面Δ=完整K=100−变体，单位为百分点。区间为未校正的探索性95%区间。"]
    rows = []
    for name, label in RETRIEVAL.items():
        if name == "full_k100": continue
        row = r["full_minus_variant"][name]
        pool, final = row["retrieval"]["candidate_known_positive_recall"], row["reranked"]["known_positive_recall@10"]
        rows.append([label, f"{100*pool['delta']:+.4f} {ci(pool['ci95'],100)}", f"{100*final['delta']:+.4f} {ci(final['ci95'],100)}"])
    table(lines, ["对照变体", "候选池召回Δ [CI]", "最终Recall@10 Δ [CI]"], rows)
    subset = r["exclusion_subset"]["full_minus_no_exclusion"]["retrieval"]["candidate_known_positive_recall"]
    base, large = r["methods"]["full_k100"], r["methods"]["full_k200"]
    lines += [f"在预先定义的{r['exclusion_subset']['queries']}个触发排除规则的查询中，开启−关闭硬过滤的候选召回差值为{100*subset['delta']:+.4f}个百分点，区间{ci(subset['ci95'],100)}。硬规则可能误解含否定词的商品名称，这是待逐查询核验的机制解释；本次直接证据是已知正例召回的变化。检索硬过滤与排序器学习的软约束必须分开。", "",
              f"候选K=100→200，候选召回{base['retrieval']['candidate_known_positive_recall']['mean']:.2%}→{large['retrieval']['candidate_known_positive_recall']['mean']:.2%}，最终Recall@10为{base['reranked']['known_positive_recall@10']['mean']:.2%}→{large['reranked']['known_positive_recall@10']['mean']:.2%}。多召回正例不自动转化为Top10收益。原始token模式同时改变停用词、否定和减号处理，不能归功于单条规则。", "",
              "耗时仅描述当前机器：并行任务负载与OS缓存未受控；检索计时包含SQL和过滤，不含查询规划。不能据此声称加速。[检索汇总](../reports/ablation-retrieval-v1/report.json)。", "",
              "## 复现与证据", "",
              "公共实验先按README准备官方数据并完成基准，再运行[特征协议](FEATURE_ABLATION_PROTOCOL.md)和[检索协议](RETRIEVAL_ABLATION_PROTOCOL.md)。无需GPU，输入/源码/模型指纹变化会拒绝评测，重跑必须用新输出目录。", "",
              "神经实验按[神经协议](NEURAL_ABLATION_PROTOCOL.md)运行。历史子集、模型权重与缓存不随源码发布，精确复现需自行取得匹配哈希的资源；仅克隆代码不能一键复现历史神经数字。不同输入应另建实验。", "",
              "表格可以从公开汇总重新生成：", "", "```bash", "python scripts/summarize_ablations.py", "```", "",
              "重画上图另需Matplotlib（不影响标准库实验入口）：", "", "```bash", "python -m pip install matplotlib", "python scripts/plot_ablations.py", "```"]
    table(lines, ["报告", "报告SHA256", "协议SHA256"], [[name, f"`{hashlib.sha256(paths[name].read_bytes()).hexdigest()}`",
          f"`{loaded[name]['protocol_sha256']}`"] for name in names])
    lines += ["[本轮验证记录](../reports/ablation-summary-v1/verification.json)包括测试与中断恢复信息。所有bootstrap只衡量查询抽样不确定性，不包含多训练种子、训练集变化、上游预训练重叠或线上用户变化。", "",
              "## 能说明的能力与下一步", "",
              "本次能展示可复现的电商学习排序、CE语义融合、查询约束取舍、检索与重排误差分离，以及保留负面结果的实验能力。多模态、个性化、生成式检索和ANN/mmap未进入这条共同消融链，不能归入这里的NDCG提升。没有千亿规模、线上转化收益或顶会成果证据。", "",
              "下一步优先在新的训练/验证划分上研究软约束或实体感知的否定处理，并用检索候选训练排序器，再用新的未见测试查询确认收益。当前不依据已见测试集挑赢家。", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines).encode("utf-8")
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("wb") as handle:
        handle.write(content); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, output)
    if output.read_bytes() != content:
        raise OSError("Report write verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=Path("docs/ABLATION_RESULTS.md"))
    args = parser.parse_args()
    render(args.reports, args.output)
    print(args.output)
