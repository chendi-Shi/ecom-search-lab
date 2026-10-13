"""Plot the published retrospective ablations; no data download or model run.

Requires matplotlib. Run from the repository root:
    python scripts/plot_ablations.py
All plotted values and intervals are read from the three report.json files.
The figure does not pool studies or add their overlapping effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


FEATURES = (
    ("title_bm25", "Title BM25"),
    ("coverage", "Token coverage"),
    ("char_trigram", "Character trigrams"),
    ("brand", "Brand features"),
    ("number", "Numeric features"),
    ("phrase", "Phrase / bigram features"),
    ("negation", "Negation features"),
)
NEURAL = (
    ("add_ce_to_lexical", "Add raw CE to lexical (13 vs 12)"),
    ("drop_ce_all", "Remove all CE features (15 vs 12)"),
    ("drop_pool_stats", "Remove both CE pool statistics"),
    ("drop_ce_raw", "Remove raw CE; keep pool statistics"),
    ("drop_percentile", "Remove CE percentile"),
    ("drop_zscore", "Remove CE z-score"),
    ("drop_bm25", "Remove title BM25"),
    ("drop_coverage", "Remove token coverage"),
    ("drop_char_trigram", "Remove character trigrams"),
    ("drop_brand", "Remove brand features"),
    ("drop_number", "Remove numeric features"),
    ("drop_phrase_bigram", "Remove phrase / bigram features"),
    ("drop_negation", "Remove negation features"),
)
RETRIEVAL = (
    ("no_brand_index_k100", "Remove brand index field"),
    ("no_exclusion_filter_k100", "Remove exclusion module"),
    ("raw_tokens_k100", "Use raw query tokens"),
    ("full_k50", "Return up to 50 candidates"),
    ("full_k200", "Return up to 200 candidates"),
)


def load_report(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def interval(row, key="ci95"):
    """Scale an observed metric difference and its interval by 100."""
    values = [float(row["delta"]), *map(float, row[key])]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("A plotted result must contain a finite delta and two interval bounds")
    delta, low, high = values
    if low > high:
        raise ValueError("Reversed confidence interval")
    return 100 * delta, 100 * low, 100 * high


def style_axis(axis, labels, title, subtitle, xlabel):
    axis.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=40)
    axis.text(0, 1.025, subtitle, transform=axis.transAxes, fontsize=10,
              color="#516274", va="bottom", linespacing=1.5)
    axis.set_yticks(range(len(labels)), labels=labels, fontsize=10.5)
    axis.set_ylim(len(labels) - .45, -.65)
    axis.axvline(0, color="#566271", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axis.grid(axis="x", color="#e4e9ee", linewidth=.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#b8c4cf")
    axis.tick_params(axis="y", length=0, pad=12)
    axis.tick_params(axis="x", colors="#516274", labelsize=10)
    axis.set_xlabel(xlabel, fontsize=10.5, labelpad=8)


def draw_interval(axis, y, values, color, marker="o", label=None):
    delta, low, high = values
    axis.hlines(y, low, high, color=color, linewidth=2, zorder=3)
    axis.vlines([low, high], y - .07, y + .07, color=color, linewidth=1.3, zorder=3)
    axis.plot(delta, y, marker=marker, markersize=6, color=color, linestyle="none",
              zorder=4, label=label)


def forest(axis, entries, color):
    limits = [0.0]
    for y, (_, row) in enumerate(entries):
        values = interval(row)
        draw_interval(axis, y, values, color)
        limits.extend(values[1:])
        axis.text(1.11, y, f"{values[0]:+.2f}", transform=axis.get_yaxis_transform(),
                  fontsize=10.5, ha="right", va="center", color=color,
                  fontfamily="DejaVu Sans Mono")
    span = max(limits) - min(limits)
    pad = max(.2, span * .07)
    axis.set_xlim(min(limits) - pad, max(limits) + pad)
    axis.text(1.11, 1.025, "Delta", transform=axis.transAxes, ha="right",
              fontsize=10, color="#516274")


def render(features_path, neural_path, retrieval_path, output, png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features, features_hash = load_report(features_path)
    neural, neural_hash = load_report(neural_path)
    retrieval, retrieval_hash = load_report(retrieval_path)
    if not features["reference_reproduction"]["passed"]:
        raise ValueError("Feature reference reproduction did not pass")
    if neural["verification"]["test_max_abs_prediction_error"] > 1e-10:
        raise ValueError("Neural reference reproduction did not pass")
    if not retrieval["baseline_reproduction"]["exact_historical_rankings_checked"]:
        raise ValueError("Retrieval historical rankings were not verified")
    if set(neural["contrasts"]) != {name for name, _ in NEURAL}:
        raise ValueError("Expected all 13 registered neural contrasts")
    if set(features["comparisons_full_minus_ablated"]["retrained"]) != {name for name, _ in FEATURES}:
        raise ValueError("Expected all seven registered retrained feature contrasts")
    if set(retrieval["full_minus_variant"]) != {name for name, _ in RETRIEVAL}:
        raise ValueError("Expected all five registered retrieval contrasts")
    feature_rows = [(label, features["comparisons_full_minus_ablated"]["retrained"][key]
                    ["fixed_pool"]["ndcg@10"]) for key, label in FEATURES]
    neural_rows = []
    for key, label in NEURAL:
        source = neural["contrasts"][key]["metrics"]["ndcg10"]
        if source["family_tests"] != 13:
            raise ValueError("Neural intervals must use the 13-comparison family")
        neural_rows.append((label, {"delta": source["delta"], "ci95": source["familywise_ci"]}))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "ecom-search-ablation-v1",
                         "axes.labelcolor": "#233244", "text.color": "#233244",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    figure, axes = plt.subplots(3, 1, figsize=(13, 18),
                                gridspec_kw={"height_ratios": [7, 13, 7]})
    figure.subplots_adjust(left=.32, right=.9, top=.845, bottom=.16, hspace=.7)
    figure.text(.06, .965, "Retrospective ablations", fontsize=25, weight="bold")
    figure.text(.06, .937, "Conditional contributions in three separate evaluation settings", fontsize=14)
    figure.text(.06, .916,
                "Previously reported test sets; no new model selection. NDCG deltas × 100; recall deltas in percentage points.",
                fontsize=11, color="#516274")

    style_axis(axes[0], [label for label, _ in feature_rows],
               "A  |  Linear ranker: seven groups removed and retrained",
               f"{features['test_queries']} queries · fixed human-judged pools · pointwise 95% query-bootstrap intervals\n"
               "Exploratory, unadjusted across seven groups and multiple metrics",
               "NDCG@10: full model minus retrained model without group (×100)")
    forest(axes[0], feature_rows, "#24678d")

    style_axis(axes[1], [label for label, _ in neural_rows],
               "B  |  Neural + lexical ranker: fixed LightGBM training",
               f"{neural['data']['queries']['test']} queries · different fixed human-judged pools · all 13 NDCG contrasts\n"
               "Bonferroni-adjusted bootstrap intervals: approximate 95% familywise coverage",
               "NDCG@10: full minus removal; first row is 13-feature minus 12-feature model (×100)")
    forest(axes[1], neural_rows, "#70449b")
    axes[1].axhline(.5, color="#d7dfe6", linewidth=.8)

    style_axis(axes[2], [label for _, label in RETRIEVAL],
               "C  |  Retrieval pipeline: frozen linear ranker, different candidate sets",
               f"{retrieval['test_queries']} queries · {retrieval['catalog_products']:,} products · incomplete relevance judgments\n"
               "Pointwise 95% query-bootstrap intervals; exploratory, unadjusted",
               "Known-positive recall: full K=100 pipeline minus variant (pp)")
    limits = [0.0]
    for y, (key, _) in enumerate(RETRIEVAL):
        contrast = retrieval["full_minus_variant"][key]
        for offset, stage, metric, color, marker, label in (
            (-.14, "retrieval", "candidate_known_positive_recall", "#268071", "o", "All returned candidates"),
            (+.14, "reranked", "known_positive_recall@10", "#bf642b", "s", "Final reranked top 10"),
        ):
            values = interval(contrast[stage][metric])
            draw_interval(axes[2], y + offset, values, color, marker, label if y == 0 else None)
            limits.extend(values[1:])
    span = max(limits) - min(limits)
    axes[2].set_xlim(min(limits) - span * .07, max(limits) + span * .07)
    axes[2].legend(loc="upper center", bbox_to_anchor=(.5, -.32), ncol=2,
                   frameon=False, fontsize=10, handletextpad=.4, columnspacing=2)

    figure.text(.06, .055,
                "Read each panel independently. Positive means the left/full configuration scores higher; negative means the variant scores higher.\n"
                "Effects overlap and must not be added. Intervals quantify query-sampling uncertainty, not training-seed or new-test generalization.\n"
                "CE = CrossEncoder. Pool statistics are derived from CE scores; dropping raw CE does not remove all neural information.",
                fontsize=10, color="#516274", linespacing=1.6)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", metadata={"Date": None,
                   "Title": "Retrospective component ablations in three separate evaluation settings",
                   "Description": "Three panels from published aggregate reports. Confidence intervals and metric scales are shown separately."})
    if png is not None:
        Path(png).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png, dpi=140)
    plt.close(figure)
    return {"figure": output.as_posix(), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "report_sha256": {"features": features_hash, "neural": neural_hash, "retrieval": retrieval_hash}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("reports/ablation-features-v1/report.json"))
    parser.add_argument("--neural", type=Path, default=Path("reports/ablation-neural-v1/report.json"))
    parser.add_argument("--retrieval", type=Path, default=Path("reports/ablation-retrieval-v1/report.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/ablation-summary-v1/ablation.svg"))
    parser.add_argument("--png", type=Path, help="Optional local PNG for visual inspection")
    args = parser.parse_args(argv)
    print(json.dumps(render(args.features, args.neural, args.retrieval, args.output, args.png), indent=2))


if __name__ == "__main__":
    main()
