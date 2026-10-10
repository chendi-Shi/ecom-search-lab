"""Run meaningful contracts and preserve a small machine-readable summary."""
import argparse
import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument("--faiss-site",type=Path)
parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"verification.json")
args=parser.parse_args()
if args.faiss_site:sys.path.append(str(args.faiss_site.resolve()))
args.output.parent.mkdir(parents=True,exist_ok=True)
start=time.perf_counter()
with args.output.with_suffix(".log").open("w",encoding="utf-8") as log:
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
summary={"tests":result.testsRun,"passed":result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),"skipped":len(result.skipped),"failures":len(result.failures),"errors":len(result.errors),"successful":result.wasSuccessful(),"seconds":time.perf_counter()-start,"python":sys.executable,"scope":"unit/integration contracts; fixture model-selection test; vector cosine and FAISS persistence checks when dependencies available"}
args.output.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary))
sys.exit(0 if result.wasSuccessful() else 1)
