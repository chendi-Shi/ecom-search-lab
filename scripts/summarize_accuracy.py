"""Render the completed, frozen accuracy evidence into docs and demo cards."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.data import save_json


def main():
    out = ROOT / "outputs/accuracy-v1"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    corpus = json.loads((out / "end_to_end_report.json").read_text(encoding="utf-8"))
    model = json.loads((out / "model.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    probe = json.loads((out / "validation_feature_probe.json").read_text(encoding="utf-8"))
    if report["status"] != "completed" or corpus["status"] != "completed":
        raise ValueError("final evaluation has not completed")
    if len(model["feature_names"]) != 15 or model["feature_names"][-2:] != ["ce_query_percentile", "ce_query_zscore"]:
        raise ValueError("summary requires the frozen 15-feature accuracy model")
    if probe.get("status") != "completed" or probe["best_spec"]["family"] != "A":
        raise ValueError("validation development record is missing or inconsistent")
    names = {"bm25": "BM25", "transferred_pairwise": "旧 pairwise 权重迁移", "cross_encoder": "CrossEncoder int8", "selected": "验证集选定的新模型"}
    m = report["methods"]
    lines = ["# v0.3：新的准确率实验", "",
        "本报告直接由冻结测试产物生成。新数据排除旧实验全部查询；旧测试成绩不能与本表直接纵向比较。", "",
        f"主评测：{report['data']['test_queries']} 个新测试查询、{report['data']['candidate_pairs']:,} 个官方人工标注 Query/商品对。"
        f"各方法对每个 Query 使用完全相同的全部已标候选，目录共 {report['data']['catalog_products']:,} 个真实商品。", "",
        "| 方法 | NDCG@10 | 前十 E/S 比例 | 首位 E 比例 | E/S Recall@10 |",
        "|---|---:|---:|---:|---:|"]
    for key, label in names.items():
        row = m[key]
        lines.append(f"| {label} | {row['ndcg10']:.4f} | {row['precision10']:.2%} | {row['exact_at1']:.2%} | {row['recall10']:.2%} |")
    lines += ["", "NDCG 使用 E/S/C/I 等级 3/2/1/0 与增益 7/3/1/0；E 表示精确匹配，S 表示可替代品。"
              "前十 E/S 比例只在这批完整人工候选中有准确率含义。", "", "## 配对不确定性", ""]
    for baseline, item in report["paired_query_bootstrap"].items():
        ci = item["ndcg10"]
        lines.append(f"- 新模型相对{names[baseline]}：NDCG 差值 {ci['delta']:+.4f}，95% 查询配对 bootstrap 区间 [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]；{ci['wins']} 胜、{ci['ties']} 平、{ci['losses']} 负。")
    for metric, label in (("exact_at1", "首位精确匹配比例"), ("precision10", "前十 E/S 比例")):
        ci = report["paired_query_bootstrap"]["transferred_pairwise"][metric]
        lines.append(f"- 相对旧模型的{label}：{100*ci['delta']:+.2f} 个百分点，95% 区间 [{100*ci['ci95'][0]:+.2f}, {100*ci['ci95'][1]:+.2f}] 个百分点。")
    lines += ["", "## 真实召回候选上的精排", "",
        f"另外对预先按查询 ID 哈希选择的 {corpus['queries']} 个新测试 Query，从完整目录召回固定 top100 候选，然后做精排。此处存在未标商品，不能把它们全部当成不相关。", "",
        "| 方法 | 已知 E/S Recall@10 | 已知 E/S Recall@100 | 前十已知正例比例 | 前十标注覆盖率 |",
        "|---|---:|---:|---:|---:|"]
    for key, row in corpus["methods"].items():
        label = names.get(key, "词法召回顺序")
        lines.append(f"| {label} | {row['known_positive_recall10']:.2%} | {row['known_positive_recall100']:.2%} | {row['known_positive_fraction10']:.2%} | {row['judged_coverage10']:.2%} |")
    ci = corpus["selected_vs_retrieval"]["known_positive_recall10"]
    lines += ["", f"相同候选池中，新精排的已知 E/S Recall@10 差值为 {ci['delta']:+.4f}，95% 区间 [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]。Recall@100 保持不变，因为本轮改的是候选排序。", "",
        "全库重排的区间跨零，当前样本不足以确认稳定的端到端收益。单独 CrossEncoder 在这项样本评测的均值更高；本轮保持测试前选定的模型，不按测试成绩更换方案。", "",
        "## 验证阶段的开发历史", "",
        "初始 13 维模型选模完成后，在尚未打开这批新测试标签时追加 A/B 两个特征方案。"
        "沿用相同训练/验证查询、CrossEncoder 分数，以及每组 8 点的叶数/轮数网格。", "",
        "| 特征方案 | 维数 | 验证 NDCG@10 |", "|---|---:|---:|",
        f"| 初始：12 词法特征 + CE 分数 | 13 | {probe['baseline_validation']['selected']['ndcg10']:.7f} |",
        f"| A：初始 + 池内 CE 百分位/标准分 | 15 | {probe['best_per_family']['A']['ndcg10']:.7f} |",
        f"| B：A + 旧 pairwise 分数/查询长度 | 17 | {probe['best_per_family']['B']['ndcg10']:.7f} |", "",
        f"验证选定 A（{probe['best_spec']['leaves']} 叶、{probe['best_spec']['iterations']} 轮），随后重新冻结特征与模型。"
        "这些验证数字受重复选模影响，不是最终泛化证据；本报告前面的冻结测试结果才承担该检验。"
        "完整网格保存在 validation_feature_probe.json，初始 model/selection/protocol/validation 和特征元数据保存在 development-initial/。", "",
        "## 模型、资源与复现", "",
        f"验证集选择 `{model['kind']}`，验证 NDCG@10={validation['methods']['selected']['ndcg10']:.4f}。"
        "训练从 2,000 条可用训练查询中按固定哈希选取 500 条；另用 200 条验证查询，最后才打开 200 条新测试查询。",
        "", "冻结 L6 CrossEncoder，128 token，CPU 两线程，oneDNN 动态 int8；最终 LambdaMART 使用 15 特征。"
        "其中 CE 百分位与标准分依赖同一 Query 的完整候选池，不能跨候选池复用。"
        "动态量化并非 float32 无损替代，完整输入 batch 参与缓存键。预训练模型原始权重约 90.9 MB，目录词法统计采用 512 商品 LRU。",
        "", f"50 查询串行评估 p50={corpus['latency_ms']['p50']:.1f} ms，p95={corpus['latency_ms']['p95']:.1f} ms。"
        "口径包括召回、编码和各对照排序，模型已加载；不是首次启动或并发吞吐指标。",
        "", "```powershell", ".\\run-accuracy.ps1", "```", "",
        "`requirements-accuracy.txt` 记录本次 CPU 环境的关键运行库精确版本；环境准备步骤见 `ACCURACY_PROTOCOL.md`。"
        "CE fingerprint 绑定实际 torch/transformers/tokenizers 版本、oneDNN 配置、模型与源码，"
        "不能通过随意升级依赖或修改冻结元数据复现原实验。交付离线核验仅证明匹配解释器下的异目录运行。", "",
        "`run-full.ps1` 保留旧 15,171 商品的完整研究版本；新准确率入口使用 82,427 商品词法召回和新精排，图文与 Beauty 推荐仍为独立数据任务。"
        "新目录的完整向量重建未完成，checkpoint 保留为可续跑实验，不属于默认服务或本次准确率收益。",
        "", "产物：`outputs/accuracy-v1/{protocol,selection,model,report,end_to_end_report}.json`，以及各方法的逐查询结果。"
        "查询文件、协议、算法源码和模型哈希在测试前冻结；测试后未重新调参。细节见 `ACCURACY_PROTOCOL.md`。",
        "", "这些是历史 ESCI 英文查询的离线结果，不代表线上转化率、购物满意度、中文搜索效果或千亿级系统。"
        "预训练语料与查询的重叠未知；选取至少 20 个已标商品的查询也限制了外推范围。", ""]
    (ROOT / "docs/ACCURACY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    summary_path = ROOT / "outputs/experiment-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["cards"] = [card for card in summary["cards"] if not card.get("task", "").startswith("ESCI v2")]
    baseline, selected = m["transferred_pairwise"], m["selected"]
    exact_ci = report["paired_query_bootstrap"]["transferred_pairwise"]["exact_at1"]
    summary["cards"] = [{"task": "ESCI v2 · 新精排", "title": "新查询 NDCG@10",
        "value": f"{baseline['ndcg10']:.4f} → {selected['ndcg10']:.4f}",
        "scope": f"200 个新测试 Query、相同全部人工候选；旧权重迁移 vs 验证选定 {model['kind']}；提升的 95% 区间高于零。"},
        {"task": "ESCI v2 · 首位结果", "title": "首位精确匹配率",
         "value": f"{baseline['exact_at1']:.1%} → {selected['exact_at1']:.1%}",
         "scope": f"同一批 {report['data']['test_queries']} 个测试 Query，人工 E 标签；差值 {100*exact_ci['delta']:+.1f} 个百分点，95% 配对区间 [{100*exact_ci['ci95'][0]:+.1f}, {100*exact_ci['ci95'][1]:+.1f}] 个百分点。"},
        {"task": "ESCI v2 · 全库重排", "title": "已知正例 Recall@10",
         "value": f"{corpus['methods']['retrieval']['known_positive_recall10']:.2%} → {corpus['methods']['selected']['known_positive_recall10']:.2%}",
         "scope": "82,427 商品库；50 测试 Query、相同召回池。差值区间跨零，尚未确认稳定收益；未标商品相关性未知。"},
        *summary["cards"]]
    save_json(summary_path, summary)
    print(json.dumps({"report": str(ROOT / "docs/ACCURACY_RESULTS.md"), "selected": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
