"""CPU generation experiment entry point; no downloads or old-output overwrites."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train genuine query-to-SID GRU retrieval on CPU")
    parser.add_argument("--data", type=Path, default=Path("data/esci"))
    parser.add_argument("--vectors", type=Path, default=Path("outputs/ann/flat"))
    parser.add_argument("--bm25-index", type=Path, default=Path("outputs/products.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("outputs/generative-v1"))
    parser.add_argument("--epochs", type=int, nargs="+", default=[30, 60, 100])
    parser.add_argument("--beams", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    from ecom_search.generative import train_experiment
    report = train_experiment(args.data, args.vectors, args.bm25_index, args.output,
                              epochs=tuple(args.epochs), beams=tuple(args.beams), threads=args.threads, seed=args.seed)
    print(json.dumps({key: report[key] for key in ("split_queries", "parameter_count", "selected_epochs", "selected_beam_width", "test", "collision", "resources")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
