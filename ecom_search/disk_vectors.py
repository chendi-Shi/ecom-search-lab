"""Bounded-workspace CPU retrieval over colocated, memory-mapped vectors.

Row-wise int8 quantization reduces backing-file bytes. Queries scan chunks and
retain only k winners; neither all-vector decoding nor an N-sized score array
is required. This prototype establishes a local quality/resource tradeoff,
not a distributed or billion-item deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import statistics
import time

from .data import load_jsonl,save_json
from .vectors import _numpy,_sha256,normalize_vectors

ROOT=Path(__file__).resolve().parents[1]


def _close_mapping(array):
    """Release the OS mapping explicitly; garbage collection is not a close API."""
    mapping=getattr(array,"_mmap",None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def build_disk_store(source: str|Path, destination: str|Path, *, mode: str="int8", chunk: int=1024) -> dict:
    if mode not in {"float32","int8"} or type(chunk) is not int or chunk<1:
        raise ValueError("mode must be float32/int8 and chunk positive")
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("disk store exists; choose a new directory")
    np=_numpy()
    matrix=target=scales=None
    try:
        matrix=np.load(source/"vectors.npy",mmap_mode="r",allow_pickle=False)
        ids=json.loads((source/"ids.json").read_text(encoding="utf-8"))
        if matrix.ndim!=2 or matrix.shape[1]<1 or len(ids)!=len(matrix) or len(ids)!=len(set(ids)):
            raise ValueError("source vectors/IDs invalid")
        metadata=json.loads((source/"metadata.json").read_text(encoding="utf-8"))
        if _sha256(source/"vectors.npy")!=metadata["vectors_sha256"] or _sha256(source/"ids.json")!=metadata["ids_sha256"]:
            raise ValueError("source checksums differ")
        destination.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter()
        filename="quantized.npy" if mode=="int8" else "vectors.npy"
        target=np.lib.format.open_memmap(destination/filename,mode="w+",dtype=np.int8 if mode=="int8" else np.float32,shape=matrix.shape)
        scales=np.lib.format.open_memmap(destination/"scales.npy",mode="w+",dtype=np.float32,shape=(len(matrix),)) if mode=="int8" else None
        for first in range(0,len(matrix),chunk):
            block=normalize_vectors(matrix[first:first+chunk])
            if mode=="int8":
                scale=np.max(np.abs(block),axis=1)/127.0
                quantized=np.clip(np.rint(block/scale[:,None]),-127,127).astype(np.int8)
                # q_int8 dot query / ||q_int8|| is cosine for reconstructed q.
                scales[first:first+len(block)]=(1.0/np.linalg.norm(quantized.astype(np.float32),axis=1)).astype(np.float32)
                target[first:first+len(block)]=quantized
            else:target[first:first+len(block)]=block
        target.flush()
        if scales is not None:scales.flush()
        _close_mapping(target)
        _close_mapping(scales)
        shutil.copyfile(source/"ids.json",destination/"ids.json")
        files=[filename,"ids.json"]+(["scales.npy"] if mode=="int8" else [])
        report={"schema_version":1,"mode":mode,"count":len(ids),"dimension":matrix.shape[1],"matrix":filename,"model":metadata.get("model"),"input_fingerprint":metadata.get("input_fingerprint"),"source_vectors_sha256":metadata["vectors_sha256"],"files":{name:{"sha256":_sha256(destination/name),"bytes":(destination/name).stat().st_size} for name in files},"build_seconds":time.perf_counter()-start,"scale_meaning":"inverse reconstructed int8 row norm" if mode=="int8" else None}
        report["backing_file_bytes"]=sum(info["bytes"] for info in report["files"].values())
        save_json(destination/"metadata.json",report)
        return report
    finally:
        for array in (target,scales,matrix):
            _close_mapping(array)


class DiskVectorIndex:
    def __init__(self,directory: str|Path,*,chunk: int=1024,verify: bool=True):
        if type(chunk) is not int or chunk<1:
            raise ValueError("chunk must be a positive integer")
        self.matrix=self.scales=self.encoder=None
        self._closed=False
        self.directory=Path(directory).resolve()
        self.metadata=json.loads((self.directory/"metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema_version")!=1 or self.metadata.get("mode") not in {"int8","float32"}:
            raise ValueError("unsupported disk store")
        if verify:
            for name,info in self.metadata["files"].items():
                if _sha256(self.directory/name)!=info["sha256"]:
                    raise ValueError("disk store checksum mismatch")
        self.np=_numpy()
        self.ids=json.loads((self.directory/"ids.json").read_text(encoding="utf-8"))
        try:
            self.matrix=self.np.load(self.directory/self.metadata["matrix"],mmap_mode="r",allow_pickle=False)
            self.scales=self.np.load(self.directory/"scales.npy",mmap_mode="r",allow_pickle=False) if self.metadata["mode"]=="int8" else None
            if self.matrix.shape!=(self.metadata["count"],self.metadata["dimension"]) or len(self.ids)!=len(self.matrix) or len(set(self.ids))!=len(self.ids):
                raise ValueError("disk store shape/IDs mismatch")
            if self.scales is not None and self.scales.shape!=(len(self.ids),):raise ValueError("scale shape mismatch")
        except BaseException:
            self.close()
            raise
        self.chunk=chunk

    def close(self):
        for array in (self.matrix,self.scales):
            _close_mapping(array)
        self.matrix=self.scales=self.encoder=None
        self._closed=True

    def __enter__(self):
        if self._closed:raise ValueError("disk vector index is closed")
        return self

    def __exit__(self,*exception):
        self.close()

    def search_vector(self,vector,k: int=100)->list[tuple[str,float]]:
        if self._closed:raise ValueError("disk vector index is closed")
        if type(k) is not int or k<0:raise ValueError("k must be a nonnegative integer")
        if not k or not self.ids:return []
        np=self.np
        query=normalize_vectors(np.asarray(vector,dtype=np.float32).reshape(1,-1))[0]
        if len(query)!=self.matrix.shape[1]:raise ValueError("query dimension differs")
        winners=[]
        for first in range(0,len(self.matrix),self.chunk):
            last=min(first+self.chunk,len(self.matrix))
            scores=self.matrix[first:last].astype(np.float32)@query
            if self.scales is not None:scores*=self.scales[first:last]
            keep=min(k,len(scores))
            above=ties=None
            if keep<len(scores):
                threshold=np.partition(scores,len(scores)-keep)[len(scores)-keep]
                above=np.flatnonzero(scores>threshold)
                ties=np.flatnonzero(scores==threshold)
                # Take boundary ties by ascending row index. Arbitrary tied
                # argpartition survivors would otherwise depend on chunk size.
                positions=np.concatenate((above,ties[:keep-len(above)]))
            else:positions=range(len(scores))
            for position in positions:
                index=first+int(position)
                item=(float(scores[position]),-index,index)
                if len(winners)<k:heapq.heappush(winners,item)
                elif item>winners[0]:heapq.heapreplace(winners,item)
            del scores,positions,above,ties
        return [(self.ids[index],score) for score,_,index in sorted(winners,reverse=True)]

    def search(self,query: str,k: int=100)->list[tuple[str,float]]:
        if self._closed:raise ValueError("disk vector index is closed")
        if not isinstance(query,str) or not query.strip():raise ValueError("query must be nonempty text")
        if self.encoder is None:
            from .vectors import _load_encoder,model_fingerprint
            model=self.metadata.get("model")
            if not model:raise ValueError("no text encoder recorded")
            model_path=Path(model["path"])
            if not model_path.is_dir():
                model_path=ROOT/"outputs/text-assets/minilm"
                if not model_path.is_dir():raise ValueError("recorded text model is missing and the packaged MiniLM copy is unavailable")
            if model_fingerprint(model_path)["sha256"]!=model["sha256"]:raise ValueError("local text model changed")
            self.encoder=_load_encoder(model_path)
        vector=self.encoder.encode([query],convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)[0]
        return self.search_vector(vector,k)

    def workspace_estimate(self,k: int=100)->dict:
        # Upper bound on explicit ndarray workspace; Python IDs/heaps and OS
        # page-cache residency are additional, and are not called measured RSS.
        rows=min(self.chunk,len(self.ids))
        # 4*rows*dim for the decoded block; 64*rows conservatively covers
        # scores, partition copy, masks and simultaneous int64 tie/position
        # arrays; 32*dim covers float32/float64 normalization temporaries.
        return {"explicit_array_workspace_upper_bound_bytes":rows*self.matrix.shape[1]*4+rows*64+self.matrix.shape[1]*32,"workspace_estimator_version":2,"workspace_formula":"4*chunk_rows*dimension + 64*chunk_rows + 32*dimension","winner_heap_max_entries":min(k,len(self.ids)),"mapped_backing_bytes":self.metadata["backing_file_bytes"],"memory_limit_scope":"conservative explicit ndarray workspace only; OS page cache, BLAS internals, interpreter, IDs and Python heap tuples are additional"}


def _percentile(values,p):
    values=sorted(values)
    position=(len(values)-1)*p
    lower=int(position)
    upper=min(lower+1,len(values)-1)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def benchmark(source: str|Path,output: str|Path,validation: str|Path,test: str|Path)->dict:
    from .vectors import VectorIndex,_load_encoder,_embeddings
    np=_numpy()
    output,source=Path(output).resolve(),Path(source).resolve()
    if (output/"report.json").exists():raise FileExistsError("report exists; use a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    builds={mode:build_disk_store(source,output/mode,mode=mode) for mode in ("float32","int8")}
    reference=VectorIndex(source)
    encoder=_load_encoder(Path(reference.metadata["model"]["path"]))
    val=load_jsonl(validation)
    if not val:raise ValueError("validation is empty")
    val_vectors,val_cache=_embeddings([row["query"] for row in val],reference.metadata["model"],output/"cache",encoder,32)
    reference_neighbors=[reference.search_vector(vector,100) for vector in val_vectors]
    choices=[]
    for chunk in (256,1024,4096):
        with DiskVectorIndex(output/"int8",chunk=chunk) as index:
            durations=[]
            recalls=[]
            for vector,gold in zip(val_vectors,reference_neighbors):
                start=time.perf_counter();hits=index.search_vector(vector,100);durations.append(1000*(time.perf_counter()-start))
                recalls.append(len({pid for pid,_ in hits[:10]}&{pid for pid,_ in gold[:10]})/10)
            choices.append({"chunk":chunk,"validation_neighbor_recall_at_10":statistics.mean(recalls),"validation_p95_ms":_percentile(durations,.95),"workspace":index.workspace_estimate()})
    chosen=min(choices,key=lambda row:(-row["validation_neighbor_recall_at_10"],row["validation_p95_ms"],row["chunk"]))
    selection={"chunk":chosen["chunk"],"policy":"maximum validation neighbor Recall@10, then minimum p95; chunk grid [256,1024,4096]","validation":choices,"test_used":False,"frozen_before_test":True}
    save_json(output/"selection.json",selection)
    rows=load_jsonl(test)
    if not rows:raise ValueError("test is empty")
    if {r["query_id"] for r in rows}&{r["query_id"] for r in val}:raise ValueError("validation/test query IDs overlap")
    vectors,test_cache=_embeddings([row["query"] for row in rows],reference.metadata["model"],output/"cache",encoder,32)
    expected=[reference.search_vector(vector,100) for vector in vectors]
    results={}
    for mode in ("float32","int8"):
        with DiskVectorIndex(output/mode,chunk=chosen["chunk"]) as index:
            samples=[];neighbor=[];recalls=[];coverage=[]
            for row,vector,gold in zip(rows,vectors,expected):
                start=time.perf_counter();hits=index.search_vector(vector,100);samples.append(1000*(time.perf_counter()-start))
                ids={pid for pid,_ in hits};positives={pid for pid,grade in row["qrels"].items() if grade>=2}
                neighbor.append(len({pid for pid,_ in hits[:10]}&{pid for pid,_ in gold[:10]})/10)
                if positives:recalls.append(len(ids&positives)/len(positives))
                coverage.append(len(ids&set(row["qrels"]))/len(ids) if ids else 0)
            results[mode]={"neighbor_recall_at_10":statistics.mean(neighbor),"known_positive_recall_at_100":statistics.mean(recalls) if recalls else None,"judged_coverage_at_100":statistics.mean(coverage),"p50_ms":_percentile(samples,.5),"p95_ms":_percentile(samples,.95),"workspace":index.workspace_estimate(),"backing_file_bytes":builds[mode]["backing_file_bytes"],"build_seconds":builds[mode]["build_seconds"],"latency_scope":"cached query vector; chunk scan and ID mapping; excludes model encoding"}
    report={"scope":"real 15,171 ESCI product-pool retrieval, incomplete per-query qrels; no billion-scale deployment claim","selection":selection,"test":results,"test_queries":len(rows),"source":str(source),"source_input_fingerprint":reference.metadata["input_fingerprint"],"data_fingerprints":{"validation":_sha256(Path(validation)),"test":_sha256(Path(test))},"builds":builds,"limitations":["mmap uses OS page cache; explicit workspace is not total resident memory","int8 chunk scan is exact over quantized vectors, not ANN","smaller backing storage can trade accuracy and CPU latency; no distributed network/storage validation"]}
    save_json(output/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=Path("outputs/ann/flat"))
    parser.add_argument("--output",type=Path,default=Path("outputs/disk-vectors-v1"))
    parser.add_argument("--validation",type=Path,default=Path("data/esci/corpus_val.jsonl"))
    parser.add_argument("--test",type=Path,default=Path("data/esci/corpus_test.jsonl"))
    parser.add_argument("--faiss-site",type=Path)
    args=parser.parse_args()
    if args.faiss_site:
        from .vectors import _faiss
        _faiss(str(args.faiss_site))
    result=benchmark(args.source,args.output,args.validation,args.test)
    print(json.dumps({"report":str(args.output/"report.json"),"test":result["test"]},indent=2))


if __name__=="__main__":main()
