from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .data import load_jsonl, save_json
from .index import ProductIndex, build_index, catalog_digest
from .query import plan_query


class SearchEngine:
    def __init__(self, index_path: Path, model_path: Path | None = None, vector_path: Path | None = None, *, generative_path: Path | None=None, disk_path: Path | None=None, profile_path: Path | None=None, accuracy_path: Path | None=None, strategy_path: Path | None=None):
        self.index = ProductIndex(index_path)
        self.model = None
        self.extractor = None
        self.vector = None
        self.generative = None
        self.disk = None
        self.accuracy = None
        self.retrieval_pool = None
        self.profile_store = None
        self.profile_vectors = None
        self.profile_positions = None
        try:
            self._load(model_path, vector_path, generative_path, disk_path, profile_path, accuracy_path, strategy_path)
        except Exception:
            self.close()
            raise

    def _load(self, model_path, vector_path, generative_path, disk_path, profile_path, accuracy_path, strategy_path):
        if model_path is not None:
            from .ranking import FeatureExtractor, PairwiseRanker
            obj = json.loads(model_path.read_text(encoding="utf-8"))
            if obj.get("catalog_sha256") != self.index.metadata["catalog_sha256"]:
                self.index.close()
                raise ValueError("model/index catalog fingerprints differ; rebuild/train on the same catalog")
            self.model = PairwiseRanker.from_dict(obj.get("ranker", obj))
            catalog = self.index.catalog()
            if catalog_digest(catalog) != self.index.metadata["catalog_sha256"]:
                self.index.close()
                raise ValueError("indexed catalog changed since build")
            self.extractor = FeatureExtractor(catalog)
        if vector_path is not None:
            from .vectors import VectorIndex, _fingerprint
            self.vector = VectorIndex(vector_path)
            catalog = self.index.catalog()
            texts = [" ".join(str(row.get(field,"")) for field in ("title","brand","description")).strip() for row in catalog]
            expected = _fingerprint({"ids":[p["id"] for p in catalog],"texts":texts,"model":self.vector.metadata["model"]["sha256"]})
            if expected != self.vector.metadata.get("input_fingerprint"):
                raise ValueError("vector/lexical catalog fingerprints differ")
        if disk_path is not None:
            from .disk_vectors import DiskVectorIndex
            self.disk=DiskVectorIndex(disk_path,chunk=1024)
            from .vectors import _fingerprint
            catalog = self.index.catalog()
            source = self.disk.metadata
            texts = [" ".join(str(row.get(field,"")) for field in ("title","brand","description")).strip() for row in catalog]
            expected = _fingerprint({"ids":[p["id"] for p in catalog],"texts":texts,"model":source["model"]["sha256"]})
            if self.disk.ids != [p["id"] for p in catalog] or expected != source.get("input_fingerprint"):
                raise ValueError("disk/lexical catalog fingerprints differ")
        if generative_path is not None:
            from .generative import GenerativeIndex
            self.generative=GenerativeIndex(generative_path)
            if self.generative.metadata.get("catalog_sha256") != self.index.metadata["catalog_sha256"]:
                raise ValueError("generative/lexical catalog fingerprints differ")
        if accuracy_path is not None or strategy_path is not None:
            catalog = self.index.catalog()
            if catalog_digest(catalog) != self.index.metadata["catalog_sha256"]:
                raise ValueError("indexed catalog changed since build")
            if strategy_path is None:
                strategy_path = Path(accuracy_path).parent / "retrieval/pool_strategy.json"
            strategy_path = Path(strategy_path)
            from scripts.prepare_accuracy_retrieval import RetrievalPool
            self.retrieval_pool = RetrievalPool(strategy_path.parent, strategy_path=strategy_path)
            if self.retrieval_pool.metadata.get("catalog_sha256") != self.index.metadata["catalog_sha256"]:
                raise ValueError("retrieval strategy/lexical catalog fingerprints differ")
            if accuracy_path is not None:
                from .accuracy import AccuracyRanker, CompactFeatureExtractor
                accuracy_extractor = CompactFeatureExtractor(catalog)
                if self.extractor is None:
                    self.extractor = accuracy_extractor
                self.accuracy = AccuracyRanker(Path(accuracy_path), accuracy_extractor,
                    cache_dir=Path(accuracy_path).parent / "serving-cache")
        if profile_path is not None:
            from .personalization import PersonalizationStore
            self.profile_store=PersonalizationStore(profile_path)

    def close(self):
        self.index.close()
        if self.profile_store:self.profile_store.close()
        if self.disk is not None:self.disk.close()
        if self.retrieval_pool is not None:self.retrieval_pool.close()
        if self.accuracy is not None:self.accuracy.close()
        if self.profile_vectors is not None:
            self.profile_vectors._mmap.close()
            self.profile_vectors=None

    def record_preference(self,user_id: str,product_id: str,action: str)->dict:
        if self.profile_store is None:raise ValueError("local profiles are not configured")
        if product_id not in self.index.get_products([product_id]):raise ValueError("unknown ESCI product ID")
        return self.profile_store.record(user_id,product_id,action)

    def _personalize(self,query: str,candidates: list[dict],user_id: str,weight: float):
        from .personalization import ProfileRanker
        if self.profile_store is None:raise ValueError("local profiles are not configured")
        if not isinstance(weight,(int,float)) or isinstance(weight,bool) or not 0<=weight<=1:
            raise ValueError("personalization_weight must be between 0 and 1")
        profile=self.profile_store.profile(user_id)
        hidden=set(profile["hidden_ids"])
        candidates=[product for product in candidates if product["id"] not in hidden]
        if not candidates or not profile["events"]:return candidates,{"applied":False,"profile":profile}
        profile_ids={row["product_id"] for row in self.profile_store.events(user_id)}|{p["id"] for p in candidates}
        catalog={}
        ordered=sorted(profile_ids)
        for first in range(0,len(ordered),1000):catalog.update(self.index.get_products(ordered[first:first+1000]))
        vectors=None
        if self.vector is not None:
            from .vectors import _numpy
            if self.profile_vectors is None:
                self.profile_vectors=_numpy().load(self.vector.directory/"vectors.npy",mmap_mode="r",allow_pickle=False)
                self.profile_positions={pid:i for i,pid in enumerate(self.vector.ids)}
            vectors={pid:self.profile_vectors[self.profile_positions[pid]] for pid in catalog}
        ranker=ProfileRanker(self.profile_store,list(catalog.values()),vectors=vectors)
        scores=ranker.score(query,candidates,user_id)
        low=min(p["score"] for p in candidates);high=max(p["score"] for p in candidates)
        for product in candidates:
            product["relevance_score"]=product["score"]
            product["preference_score"]=scores[product["id"]]
            relevance=(product["score"]-low)/(high-low) if high>low else .5
            product["score"]=(1-weight)*relevance+weight*(scores[product["id"]]+1)/2
        candidates.sort(key=lambda p:(-p["score"],p["id"]))
        return candidates,{"applied":True,"backend":ranker.backend,"weight":weight,"profile":profile,"scope":"explicit local preferences; separate from the independently trained Beauty BPR experiment"}

    def _retrieve(self, query, plan, pool, use_generative):
        if self.retrieval_pool is not None:
            # This helper owns the exact validation-selected fusion, raw query
            # encoding and exclusion behavior. Do not fuse/filter a second time.
            hits = self.retrieval_pool.search(query, k=pool, use_vector=True)
            ids = [pid for pid, _ in hits]
            if len(ids) > pool or len(ids) != len(set(ids)):
                raise ValueError("retrieval strategy returned duplicate or excessive candidates")
            products = self.index.get_products(ids)
            if set(products) != set(ids):
                raise ValueError("retrieval strategy returned unknown product IDs")
            converted = [dict(products[pid], retrieval_score=float(score)) for pid, score in hits]
            if any(not math.isfinite(p["retrieval_score"]) for p in converted):
                raise ValueError("retrieval strategy returned nonfinite scores")
            mode = getattr(self.retrieval_pool, "selection", {}).get("mode")
            return converted, "fts_bm25" if mode == "lexical_only" else "validation_selected_retrieval"
        candidates = self.index.search(plan,pool)
        additional=[]
        retrieval_channels=["fts"]
        retrieval_k=min(1000,pool*3) if plan.excluded_terms else pool
        if self.vector is not None:
            additional.append(self.vector.search(" ".join(plan.positive_terms),retrieval_k))
            retrieval_channels.append(self.vector.metadata["algorithm"])
        if self.disk is not None:
            additional.append(self.disk.search(" ".join(plan.positive_terms),retrieval_k))
            retrieval_channels.append("disk_"+self.disk.metadata["mode"])
        if self.generative is not None and use_generative:
            additional.append(self.generative.search(query,retrieval_k))
            retrieval_channels.append("generative_sid")
        if additional:
            fusion = {}
            for rank,product in enumerate(candidates,1):
                fusion[product["id"]] = fusion.get(product["id"],0)+1/(60+rank)
            for channel in additional:
                for rank,(pid,score) in enumerate(channel,1):
                    fusion[pid] = fusion.get(pid,0)+1/(60+rank)
            ids = sorted(fusion,key=lambda pid:(-fusion[pid],pid))
            products={}
            for first in range(0,len(ids),1000):products.update(self.index.get_products(ids[first:first+1000]))
            if set(products)!=set(ids):raise ValueError("retrieval channel returned unknown product IDs")
            candidates = [dict(products[pid],retrieval_score=fusion[pid]) for pid in ids]
            if plan.excluded_terms:
                from .ranking import FeatureExtractor
                forbidden = set(plan.excluded_terms)
                candidates = [p for p in candidates if not forbidden & FeatureExtractor.affirmed_product_terms(p)]
            candidates = candidates[:pool]
        return candidates, "_".join(retrieval_channels)+("_rrf" if additional else "_bm25")

    def search(self, query: str, k: int = 5, pool: int = 100, *,user_id: str | None=None,personalization_weight: float=.25,use_generative: bool=False,use_accuracy: bool | None=None) -> dict:
        if not isinstance(use_generative,bool):raise ValueError("use_generative must be a boolean")
        if use_accuracy is not None and not isinstance(use_accuracy,bool):raise ValueError("use_accuracy must be a boolean")
        enabled = self.accuracy is not None if use_accuracy is None else use_accuracy
        if enabled and self.accuracy is None:
            raise ValueError("accuracy model is not configured")
        if not isinstance(k,int) or isinstance(k,bool) or not 1 <= k <= 100:
            raise ValueError("k must be an integer in [1, 100]")
        if not isinstance(pool,int) or isinstance(pool,bool) or not k <= pool <= 1000:
            raise ValueError("pool must be an integer in [k, 1000]")
        if (enabled or self.retrieval_pool is not None) and pool > 200:
            raise ValueError("accuracy/frozen retrieval pool must be at most 200 (default 100)")
        if use_generative and self.retrieval_pool is not None:
            raise ValueError("generative fusion is unavailable with the frozen accuracy retrieval strategy")
        start = time.perf_counter()
        plan = plan_query(query)
        candidates, retrieval = self._retrieve(query, plan, pool, use_generative)
        ranking = "retrieval_order"
        if enabled:
            # Query text must match training and validation; the joint encoder
            # receives the original query, not only positive lexical tokens.
            scores = self.accuracy.score(query, candidates)
            if set(scores) != {p["id"] for p in candidates} or any(
                    isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                    for score in scores.values()):
                raise ValueError("accuracy model returned missing or nonfinite candidate scores")
            for product in candidates:
                product["score"] = float(scores[product["id"]])
            candidates.sort(key=lambda p:(-p["score"],p["id"]))
            ranking = "cpu_" + self.accuracy.metadata["kind"]
        elif self.model:
            for product in candidates:
                features = self.extractor.features(plan.ranking_query,product)
                product["score"] = self.model.score(features)
                product["feature_values"] = dict(zip(self.extractor.names,features))
            candidates.sort(key=lambda p:(-p["score"],p["id"]))
            ranking = "trained_pairwise"
        else:
            for product in candidates:
                product["score"] = product["retrieval_score"]
        personalization={"applied":False}
        if user_id is not None:candidates,personalization=self._personalize(query,candidates,user_id,personalization_weight)
        return {"query":query,"plan":plan.to_dict(),"retrieval":retrieval,"ranking":ranking,
                "candidate_source":"词法召回（BM25）" if retrieval == "fts_bm25" else "验证集选定召回策略" if retrieval == "validation_selected_retrieval" else "多路召回融合",
                "accuracy":{"available":self.accuracy is not None,"applied":enabled,
                            "kind":self.accuracy.metadata["kind"] if self.accuracy is not None else None,
                            "pool_limit":200,"evaluated_pool":100 if self.accuracy is not None else None},
                "personalization":personalization,"candidate_count":len(candidates),"products":candidates[:k],"latency_ms":1000*(time.perf_counter()-start),"data":"ESCI historical product text; no current price or availability data"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU ecommerce search on frozen Amazon ESCI")
    sub = parser.add_subparsers(dest="command",required=True)
    build = sub.add_parser("build-index")
    build.add_argument("--catalog",type=Path,default=Path("data/esci/catalog.jsonl"))
    build.add_argument("--index",type=Path,default=Path("outputs/products.sqlite"))
    train = sub.add_parser("train-evaluate")
    train.add_argument("--data",type=Path,default=Path("data/esci"))
    train.add_argument("--output",type=Path,default=Path("outputs/ranking"))
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--index",type=Path,default=Path("outputs/products.sqlite"))
    search.add_argument("--model",type=Path)
    search.add_argument("--vector-index",type=Path)
    search.add_argument("--generative-index",type=Path)
    search.add_argument("--use-generative",action="store_true")
    search.add_argument("--disk-index",type=Path)
    search.add_argument("--accuracy-model",type=Path)
    search.add_argument("--retrieval-strategy",type=Path)
    search.add_argument("--no-accuracy",action="store_true",help="Skip CPU joint-encoder reranking and use the fast candidate order")
    search.add_argument("--profiles",type=Path)
    search.add_argument("--user")
    search.add_argument("--k",type=int,default=5)
    search.add_argument("--pool",type=int,default=100)
    benchmark = sub.add_parser("benchmark-index")
    benchmark.add_argument("--index",type=Path,default=Path("outputs/products.sqlite"))
    benchmark.add_argument("--queries",type=Path,default=Path("data/esci/corpus_test.jsonl"))
    benchmark.add_argument("--output",type=Path,default=Path("outputs/retrieval.json"))
    serve = sub.add_parser("serve")
    serve.add_argument("--index",type=Path,default=Path("outputs/products.sqlite"))
    serve.add_argument("--model",type=Path)
    serve.add_argument("--vector-index",type=Path)
    serve.add_argument("--generative-index",type=Path)
    serve.add_argument("--disk-index",type=Path)
    serve.add_argument("--accuracy-model",type=Path)
    serve.add_argument("--retrieval-strategy",type=Path)
    serve.add_argument("--profiles",type=Path,default=Path("outputs/local-profiles.sqlite"))
    serve.add_argument("--multimodal-index",type=Path)
    serve.add_argument("--beauty-experiment",type=Path)
    serve.add_argument("--port",type=int,default=18866)
    args = parser.parse_args(argv)
    if args.command == "build-index":
        result = build_index(load_jsonl(args.catalog),args.index)
        save_json(args.index.with_suffix(".build.json"),result)
    elif args.command == "train-evaluate":
        from .evaluate import train_and_evaluate
        result = train_and_evaluate(args.data,args.output)
    elif args.command == "search":
        engine = SearchEngine(args.index,args.model,args.vector_index,generative_path=args.generative_index,disk_path=args.disk_index,profile_path=args.profiles,accuracy_path=args.accuracy_model,strategy_path=args.retrieval_strategy)
        try:
            result = engine.search(args.query,args.k,args.pool,user_id=args.user,use_generative=args.use_generative,use_accuracy=False if args.no_accuracy else None)
        finally:
            engine.close()
    elif args.command == "benchmark-index":
        from .resources import benchmark_index
        if args.output.exists():
            raise FileExistsError("Retrieval report exists; choose a new output path")
        result = benchmark_index(args.index,load_jsonl(args.queries))
        save_json(args.output,result)
    else:
        from .server import serve_local
        serve_local(args.index,args.model,args.port,args.vector_index,generative_path=args.generative_index,disk_path=args.disk_index,profile_path=args.profiles,multimodal_path=args.multimodal_index,beauty_path=args.beauty_experiment,accuracy_path=args.accuracy_model,strategy_path=args.retrieval_strategy)
        return
    if args.command == "train-evaluate":
        result = {"report":str(args.output/"report.md"),"test":{name:row["metrics"] for name,row in result["test"].items()},"delta":result["test_delta_learned_minus_baseline"]}
    elif args.command == "benchmark-index":
        result = {key:value for key,value in result.items() if key!="per_query"}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
