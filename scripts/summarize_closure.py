"""Verify a completed closure release and export aggregate public evidence only.

Never decodes query records or predictions. Dataset/model/index/transcript bytes
are hashed for provenance. Training metadata is read but its coefficients are
excluded by an allowlist. This command does not evaluate or select models.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.data import file_sha256
from scripts import benchmark_closure as experiment
from scripts.prepare_public_esci import canonical_sha, fingerprint, write_json

PRIMARY = "known_positive_recall@10"
COVERAGE = "candidate_recall@100"
JUDGED = "judged_fraction@10"
COPY_RUN = ("protocol.json", "report.json", "selection.json", "release.json", "validation.json", "trained.json")
RELEASE_FILES = ("protocol.json", "trained.json", "models.json", "training.json", "selection.json",
                 "validation.json", "products.sqlite", "report.json")
FORBIDDEN_KEYS = {"weights", "query", "qrels", "candidate_ids", "ranking", "query_text", "normalized_queries"}
LABELS = {"legacy_bm25": "旧硬过滤 + BM25", "legacy_judged": "旧硬过滤 + 人工候选训练",
          "legacy_retrieved": "旧硬过滤 + 召回交集训练", "soft_bm25": "取消自然语言硬过滤 + BM25",
          "soft_judged": "取消自然语言硬过滤 + 人工候选训练", "soft_retrieved": "取消自然语言硬过滤 + 召回交集训练"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(directory, mapping, required, label):
    require(isinstance(mapping, dict) and set(mapping) == set(required), f"Incomplete {label} fingerprint manifest")
    for name, digest in mapping.items():
        require(file_sha256(Path(directory) / name) == digest, f"Changed {label}: {name}")


def public_json(value):
    """Fail closed if an aggregate schema accidentally acquires sensitive rows."""
    if isinstance(value, dict):
        require(not FORBIDDEN_KEYS.intersection(value), "Raw query, labels, rankings or model weights in public evidence")
        for item in value.values():
            public_json(item)
    elif isinstance(value, list):
        for item in value:
            public_json(item)
    elif isinstance(value, str):
        require(not re.search(r"(?:^|\s)[A-Za-z]:[\\/]", value), "Local absolute path in public evidence")
    elif isinstance(value, float):
        require(math.isfinite(value), "Non-finite public evidence value")


def check_comparison(comparison, config):
    require(comparison.get("samples") == config["bootstrap"]["samples"] and
            comparison.get("seed") == config["bootstrap"]["seed"], "Changed bootstrap configuration")
    bounds = comparison.get("ci95")
    if bounds is None:
        require(comparison.get("queries") == 0 and comparison.get("delta") is None, "Invalid undefined comparison")
    else:
        require(isinstance(bounds, list) and len(bounds) == 2, "Invalid interval")
        values = [comparison.get("delta"), *bounds]
        require(all(type(x) in (float, int) and math.isfinite(x) and -1 <= x <= 1 for x in values), "Invalid recall difference")
        require(bounds[0] <= bounds[1] and comparison.get("queries", 0) > 0, "Invalid interval order or count")


def verify(data, run):
    """Validate the chain using opaque file hashes, without decoding query data."""
    data, run = Path(data), Path(run)
    config = experiment.check(data, run)
    manifest = read(data / "manifest.json")
    content = dict(manifest)
    digest = content.pop("manifest_content_sha256", None)
    require(digest == canonical_sha(content), "Data manifest content digest mismatch")
    require(manifest.get("protocol") == "closure-esci-v1", "Expected a fresh closure dataset")
    exclusion = manifest["previous_experiment_exclusion"]
    require(exclusion["selected_overlap_by_id"] == exclusion["selected_overlap_by_normalized_query"] == 0,
            "Historical query identity overlap")
    require(config["previous_experiment_exclusion"] == exclusion, "Protocol freshness binding differs")
    for name, expected in manifest["files"].items():
        require(fingerprint(data / name) == expected, f"Dataset output changed: {name}")
    trained, selection = read(run / "trained.json"), read(run / "selection.json")
    bound(run, trained.get("artifacts"), experiment.TRAINED_FILES, "trained")
    bound(run, selection.get("artifacts"), experiment.SELECTED_FILES, "selection")
    validation, report, release = (read(run / name) for name in ("validation.json", "report.json", "release.json"))
    bound(run, release.get("artifacts"), RELEASE_FILES, "release")
    selected, eligible = experiment.select_variant(validation["summary"])
    require(set(validation["summary"]) == set(experiment.VARIANTS), "Missing validation arm")
    require(selected == selection.get("selected") == validation.get("selected") and
            validation.get("eligible") == eligible, "Frozen validation winner differs")
    for obj in (trained, selection, report, release):
        require(obj.get("protocol") == experiment.PROTOCOL, "Evidence protocol mismatch")
    require(selection.get("test_labels_decoded") is False and trained.get("test_labels_decoded") is False,
            "Test was not sealed during training/selection")
    require(selection.get("baseline") == experiment.BASELINE and selection.get("pool") == experiment.POOL and
            selection.get("variant") == experiment.VARIANTS[selected], "Changed selected policy")
    require(report.get("protocol_sha256") == file_sha256(run / "protocol.json") and
            report.get("selection_sha256") == file_sha256(run / "selection.json"), "Report freeze SHA binding differs")
    require(report.get("prediction_sha256") == file_sha256(run / "test_transcripts.json"), "Prediction SHA binding differs")
    require(report.get("validation") == validation and report.get("no_test_retuning") is True,
            "Report selection evidence differs")
    for obj in (report, release):
        for key in ("data_fingerprints", "source_fingerprints"):
            require(obj.get(key) == config[key], f"Changed {key}")
    require(set(report["summary"]) == {experiment.BASELINE, selected}, "Final report contains unselected test arms")
    require(report["queries"] == manifest["queries"]["test"] and report["products"] == manifest["products"] and
            report["judged_pairs"] == manifest["judged_candidates"]["test"], "Report population differs")
    require(validation["queries"] == manifest["queries"]["val"], "Validation population differs")
    primary, coverage = report["primary_comparison"], report["candidate_coverage_comparison"]
    for comparison, metric in ((primary, PRIMARY), (coverage, COVERAGE)):
        check_comparison(comparison, config)
        means = [report["summary"][name][metric]["mean"] for name in (experiment.BASELINE, selected)]
        if None not in means:
            require(abs((means[1] - means[0]) - comparison["delta"]) < 1e-12, "Comparison delta disagrees with reported means")
    decision = experiment.release_decision(selected, primary, coverage)
    require(report["decision"] == decision and all(release.get(k) == v for k, v in decision.items()), "Release gate decision differs")
    require(release.get("variant") == experiment.VARIANTS[decision["serving_variant"]] and release.get("pool") == experiment.POOL,
            "Serving policy differs from release gate")
    require(trained.get("catalog_sha256") == selection.get("catalog_sha256") == release.get("catalog_sha256"),
            "Catalog bindings differ")
    require(read(run / "test_started.json").get("selection_sha256") == file_sha256(run / "selection.json"),
            "Final attempt marker does not bind selection")
    require(read(run / "selection_started.json").get("trained_sha256") == file_sha256(run / "trained.json"),
            "Selection attempt marker does not bind training")
    require(read(run / "training_started.json").get("protocol_sha256") == file_sha256(run / "protocol.json"),
            "Training attempt marker does not bind protocol")
    for name in COPY_RUN:
        public_json(read(run / name))
    public_json(manifest)
    return config, manifest, trained, selection, validation, report, release


def training_summary(training):
    fit_keys = ("epochs", "seed", "learning_rate", "l2", "training_records", "queries_with_pairs", "pairs",
                "pair_count", "max_pairs_per_query", "initial_loss", "final_loss", "loss_history", "feature_version", "feature_names")
    result = {key: training[key] for key in ("training", "retrieved_coverage", "feature_pairs", "index", "total_seconds", "python", "system", "cost_scope")}
    result["models"] = {name: {"seconds": value["seconds"], "fit": {key: value["fit"][key] for key in fit_keys}}
                        for name, value in training["models"].items()}
    result["export_note"] = "Model weights excluded; only pair counts, coverage, loss and cost retained"
    public_json(result)
    return result


def attempts(previous_run, run):
    current = read(run / "protocol.json")
    result = {"actual_completed_run": run.name, "completed_protocol_sha256": file_sha256(run / "protocol.json"),
              "fresh_clone_default_run": "closure-v1", "previous_attempts": []}
    if previous_run is None:
        return result
    previous_run = Path(previous_run)
    require(previous_run.resolve() != run.resolve(), "Previous attempt equals completed run")
    absent = ("trained.json", "models.json", "training.json", "selection_started.json", "validation.json",
              "selection.json", "test_started.json", "report.json", "release.json")
    require(not any((previous_run / name).exists() for name in absent), "Previous attempt has further evidence; do not silently omit results")
    previous = read(previous_run / "protocol.json")
    previous_sha = file_sha256(previous_run / "protocol.json")
    require(read(previous_run / "training_started.json").get("protocol_sha256") == previous_sha, "Aborted attempt protocol mismatch")
    differences = {key: {"before": previous.get(key), "after": current.get(key)}
                   for key in set(previous) | set(current) if previous.get(key) != current.get(key)}
    require(set(differences) <= {"registered_at_utc", "source_fingerprints"}, "Experimental recipe changed between attempts")
    old_sources, new_sources = previous["source_fingerprints"], current["source_fingerprints"]
    changed_sources = {key: {"before": old_sources.get(key), "after": new_sources.get(key)}
                       for key in set(old_sources) | set(new_sources) if old_sources.get(key) != new_sources.get(key)}
    require(set(changed_sources) <= {"scripts/benchmark_closure.py"}, "Unexpected frozen-source change between attempts")
    result["previous_attempts"].append({"run": previous_run.name, "status": "stopped_before_validation_or_test",
        "protocol_sha256": previous_sha, "training_started_file": fingerprint(previous_run / "training_started.json"),
        "reason": "Integrity guard fix; operator stopped the first attempt at feature caching, before validation/test.",
        "persistent_evidence": "protocol + training-start marker + index; no trained/validation/selection/test/report/release artifacts",
        "absent_artifacts_checked": list(absent), "changed_sources": changed_sources,
        "recipe_data_and_selection_rule_unchanged": True})
    public_json(previous)
    public_json(read(previous_run / "training_started.json"))
    public_json(result)
    return result


def serving_evidence(run, report, release):
    path = run / "serving-verification.json"
    if not path.exists():
        return None
    evidence = read(path)
    require(evidence.get("protocol") == experiment.PROTOCOL and evidence.get("passed") is True,
            "HTTP verification did not pass")
    require(evidence.get("queries") == report["queries"] and evidence.get("rankings") == 2 * report["queries"],
            "HTTP verification does not cover every test query/column")
    require(all(evidence.get(k) is True for k in ("candidate_order_exact", "ranking_ids_and_scores_exact", "displayed_top10_exact"))
            and evidence.get("new_metrics_computed") is False and evidence.get("decision_changed") is False,
            "HTTP replay was not exact or changed the decision")
    bindings = {"model_sha256": file_sha256(run / "models.json"), "report_sha256": file_sha256(run / "report.json"),
                "release_sha256": file_sha256(run / "release.json"), "service_sha256": file_sha256(ROOT / "scripts/serve_closure.py"),
                "serving_variant": release["serving_variant"], "pool": experiment.POOL}
    require(evidence.get("bindings") == bindings and evidence.get("test_transcripts_sha256") == report["prediction_sha256"],
            "HTTP verification release binding differs")
    require(evidence.get("verification_source_sha256") == file_sha256(ROOT / "scripts/verify_closure_serving.py")
            and evidence.get("html_sha256") == file_sha256(ROOT / "web/closure.html"), "HTTP verifier/UI source changed")
    public_json(evidence)
    return evidence


def value(summary, variant, metric):
    return summary[variant][metric]["mean"]


def pct(number):
    return "未定义" if number is None else f"{number * 100:.3f}%"


def pp(number):
    return "未定义" if number is None else f"{number * 100:+.3f}"


def interval(bounds):
    return "未定义" if bounds is None else f"[{bounds[0] * 100:+.3f}, {bounds[1] * 100:+.3f}]"


def validation_contrasts(validation):
    contrasts = []
    for ranker in ("bm25", "judged", "retrieved"):
        contrasts.append(("取消自然语言硬过滤，固定" + ranker, "legacy_" + ranker, "soft_" + ranker))
    for policy in ("legacy", "soft"):
        contrasts.append(("人工候选训练→召回交集训练，固定" + policy, policy + "_judged", policy + "_retrieved"))
    return [{"comparison": label, "before": before, "after": after,
             "deltas": {metric: value(validation["summary"], after, metric) - value(validation["summary"], before, metric)
                        for metric in (PRIMARY, COVERAGE, JUDGED)}} for label, before, after in contrasts]


def markdown(manifest, validation, report, training, bookkeeping, serving=None):
    selected, baseline = report["decision"]["selected_variant"], experiment.BASELINE
    promoted, serving_variant = report["decision"]["promoted"], report["decision"]["serving_variant"]
    lines = ["# 从消融到可验证的修正：Closure v1", "",
        f"本次已完成独立查询队列上的训练、验证集选型、一次最终测试和服务发布决策。验证集选择 **`{selected}`**；"
        + (f"最终门槛通过，服务采用 **`{serving_variant}`**。" if promoted else f"最终门槛未通过，服务保持 **`{serving_variant}`**。"), "",
        "完成流程并不自动意味着准确率提高；下表保留最终实际结果，不能把验证集收益当作测试集结论。", "",
        "## 新数据与一次性测试", "",
        f"使用 {manifest['products']:,} 件商品，{manifest['queries']['train']} / {manifest['queries']['val']} / {manifest['queries']['test']} 个 train / validation / test query。"
        f"最终 test 含 {report['judged_pairs']:,} 个原始人工判断对。新队列与此前3305个query的官方ID及归一化文本交集均为零；"
        "商品和语义改写不保证互斥，预训练语料也不在此排除证明内。[数据协议](CLOSURE_DATA.md)。", "",
        f"每个 query 的已知正例为 E/S（grade≥2）。无已知正例的 query 有 {report['queries_without_known_positives']} 个，其 Recall 未定义且不进入Recall均值；仍保留在总体与缺失计数。"
        "这是有限人工判断商品并集上的已知正例召回，未标注商品保持未知，不能视为负例。judged@10 是判断覆盖率，不是相关性Precision。", "",
        "## 验证集：固定六个配置", "",
        "每个配置均使用自己的召回池，pool=100。验证规则先要求候选Recall@100不低于基线均值，再按Recall@10取最高；相同分数按固定顺序、基线优先。", "",
        "| 配置 | 已知正例Recall@10 | 候选Recall@100 | judged@10 | 候选覆盖约束 |", "|---|---:|---:|---:|---|"]
    for name in experiment.VARIANTS:
        lines.append(f"| {LABELS[name]} (`{name}`) | {pct(value(validation['summary'], name, PRIMARY))} | {pct(value(validation['summary'], name, COVERAGE))} | {pct(value(validation['summary'], name, JUDGED))} | {'满足' if name in validation['eligible'] else '不满足'} |")
    lines += ["", "验证集内的模块差值如下，单位为百分点。它们参与选型，属于探索性解释，没有独立测试确认，也不计算或宣称可相加的贡献比例。", "",
              "| 对照（后−前） | ΔRecall@10 | Δ候选Recall@100 | Δjudged@10 |", "|---|---:|---:|---:|"]
    for row in validation_contrasts(validation):
        lines.append(f"| {row['comparison']} | {pp(row['deltas'][PRIMARY])} | {pp(row['deltas'][COVERAGE])} | {pp(row['deltas'][JUDGED])} |")
    lines += ["", "`soft` 取消自然语言否定的硬过滤，只有显式 `-word` 保留字面硬过滤；学习排序器仍可使用否定特征。"
              "`soft_bm25` 只使用BM25排序，不使用否定排序特征。这些配置都不是语义约束保证。"
              "`retrieved` 只使用 soft top100 与已有人工qrels的交集训练；未标注召回商品不充当负例。改变训练池也会改变可用训练对数量，因此不能将其效果解释为单一特征的因果贡献。", "",
              "## 最终测试：只比较冻结赢家与基线", "", "| 方法 | 已知正例Recall@10 | 候选Recall@100 | judged@10 |", "|---|---:|---:|---:|"]
    for name in dict.fromkeys((baseline, selected)):
        lines.append(f"| `{name}` | {pct(value(report['summary'], name, PRIMARY))} | {pct(value(report['summary'], name, COVERAGE))} | {pct(value(report['summary'], name, JUDGED))} |")
    lines += ["", "| 冻结赢家−基线 | 差值（百分点） | 配对95%区间（百分点） |", "|---|---:|---:|"]
    for label, comparison in (("主要指标 Recall@10", report["primary_comparison"]), ("候选覆盖 Recall@100", report["candidate_coverage_comparison"])):
        lines.append(f"| {label} | {pp(comparison['delta'])} | {interval(comparison['ci95'])} |")
    lines += ["", "区间为 query 级配对 percentile bootstrap，2000次、seed=20261013。最终仅有一个事先冻结的主要比较，候选覆盖是联合发布约束。"
              "区间反映查询抽样不确定性，不覆盖训练随机种子、其他商品库或线上流量变化。", "",
              "发布门槛是：主要指标Δ的95%下界严格大于0；同时候选Recall@100的Δ下界至少−1个百分点。选择基线本身也不会被记成新方法晋级。", "",
              f"实际决定：**{'晋级' if promoted else '不晋级'}**；服务配置 `{serving_variant}`。原因码：`{', '.join(report['decision']['reasons']) or 'none'}`。"
              "服务读取冻结release并复核指纹；未通过门槛的实验模型仍留在报告，不作为默认服务。", "",
              "## 训练范围和成本", "", "| 训练候选来源 | 可形成偏好对的query | 采样偏好对数 | 初始→最终loss | 拟合耗时（秒） |", "|---|---:|---:|---:|---:|"]
    for name, item in training["models"].items():
        fit = item["fit"]
        lines.append(f"| {name} | {fit['queries_with_pairs']} | {fit['pairs']:,} | {fit['initial_loss']:.6f}→{fit['final_loss']:.6f} | {item['seconds']:.2f} |")
    cov = training["retrieved_coverage"]
    lines += ["", f"训练召回共 {cov['retrieved_products']:,} 个商品位置，其中 {cov['judged_products']:,} 已判断、{cov['unjudged_ignored']:,} 未判断并忽略；"
              f"{cov['queries_without_pairs']} 个query无法形成训练对，{cov['queries_with_no_judged_hits']} 个没有已判断召回商品。总训练阶段耗时 {training['total_seconds']:.2f} 秒。"
              "耗时是单CPU进程在未控制主机负载下的诊断值，不是生产容量或延迟承诺。", "",
              "## 执行记录与复现", "",
              f"本次完成目录为 `outputs/{bookkeeping['actual_completed_run']}`。新克隆使用默认 `outputs/closure-v1` 即可；目录编号是执行记录，不是数据或方法版本。"]
    if bookkeeping["previous_attempts"]:
        previous = bookkeeping["previous_attempts"][0]
        lines += ["", f"第一次 `outputs/{previous['run']}` 在特征缓存阶段停止，以修正完整性校验。该目录保留；没有 trained、validation、selection、test-start、report 或 release 产物。"
                  "更改的是benchmark完整性检查，数据、模型训练参数、六个配置、验证规则和测试门槛相同；没有读过该队列的验证/测试结果后重选配方。", "",
                  f"- 首次协议SHA-256：`{previous['protocol_sha256']}`", f"- 完成协议SHA-256：`{bookkeeping['completed_protocol_sha256']}`"]
    if serving is not None:
        lines += ["", f"真实loopback HTTP已回放全部 {serving['queries']} 个最终query、两列共 {serving['rankings']} 份排序："
                  f"{serving['candidate_id_entries']:,} 个候选ID位置、{serving['score_entries']:,} 个排序分数位置及显示Top10均与离线封存预测完全一致。"
                  "此次只验证服务一致性，未重算准确率或改变选择。[HTTP一致性证据](../reports/closure-v1/serving-verification.json)。"]
    else:
        lines += ["", "此报告导出时尚未附带真实HTTP回放证据；服务实现共用离线路径，但尚不能据此宣称已完成端到端HTTP一致性验证。"]
    lines += ["", "[完整复现步骤](CLOSURE_PROTOCOL.md)；[机器可读报告](../reports/closure-v1/report.json)、"
              "[验证集六配置](../reports/closure-v1/validation.json)、[发布决定](../reports/closure-v1/release.json)、"
              "[执行记录](../reports/closure-v1/attempts.json)、[证据文件指纹](../reports/closure-v1/evidence.json)。", "",
              "本次补齐的是“发现问题→固定修正候选→新查询验证→冻结→一次测试→服务采用或回退”的流程。"
              "尚未证明千亿级检索、多模态个性化排序、线上转化收益或分布式资源优化；不能据此声称覆盖整个JD。", ""]
    return "\n".join(lines)


def generate(data, run, output, doc, previous_run=None):
    data, run, output, doc = map(Path, (data, run, output, doc))
    config, manifest, _, selection, validation, report, release = verify(data, run)
    training = training_summary(read(run / "training.json"))
    bookkeeping = attempts(previous_run, run)
    serving = serving_evidence(run, report, release)
    summary = {"protocol": experiment.PROTOCOL, "queries": manifest["queries"], "products": manifest["products"],
        "validation": validation["summary"], "validation_contrasts_exploratory": validation_contrasts(validation),
        "selected": selection["selected"], "final_summary": report["summary"],
        "primary_comparison": report["primary_comparison"], "candidate_coverage_comparison": report["candidate_coverage_comparison"],
        "decision": report["decision"], "no_model_evaluation_performed_by_exporter": True}
    public_json(summary)
    rendered = markdown(manifest, validation, report, training, bookkeeping, serving)
    require(not output.exists(), "Public evidence output already exists; preserve previous exports")
    output.mkdir(parents=True, exist_ok=False)
    for name in COPY_RUN:
        (output / name).write_bytes((run / name).read_bytes())
    for name in ("training_started.json", "selection_started.json", "test_started.json"):
        public_json(read(run / name))
        (output / name).write_bytes((run / name).read_bytes())
    (output / "data-manifest.json").write_bytes((data / "manifest.json").read_bytes())
    if previous_run is not None:
        (output / "aborted-protocol.json").write_bytes((Path(previous_run) / "protocol.json").read_bytes())
        (output / "aborted-training-started.json").write_bytes((Path(previous_run) / "training_started.json").read_bytes())
    if serving is not None:
        (output / "serving-verification.json").write_bytes((run / "serving-verification.json").read_bytes())
    write_json(output / "training-summary.json", training)
    write_json(output / "attempts.json", bookkeeping)
    write_json(output / "summary.json", summary)
    evidence = {"verification": "passed before export", "exporter_sha256": file_sha256(Path(__file__)),
        "raw_query_records_models_indexes_transcripts_exported": False,
        "source_training_sha256": file_sha256(run / "training.json"),
        "files": {path.name: fingerprint(path) for path in sorted(output.iterdir())},
        "verified_bindings": ["protocol/source/data", "trained artifacts", "validation selection artifacts", "report protocol/selection/prediction",
                              "release artifacts", "release gate", "exclusive stage-start markers", "historical attempt status"]}
    write_json(output / "evidence.json", evidence)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(rendered, encoding="utf-8", newline="\n")
    return {"status": "verified_aggregate_evidence_exported", "serving_variant": report["decision"]["serving_variant"],
            "promoted": report["decision"]["promoted"], "files": len(list(output.iterdir()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/esci-closure-v1")
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/closure-v1")
    parser.add_argument("--previous-run", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/closure-v1")
    parser.add_argument("--doc", type=Path, default=ROOT / "docs/CLOSURE_RESULTS.md")
    args = parser.parse_args()
    print(json.dumps(generate(args.data, args.run, args.output, args.doc, args.previous_run), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
