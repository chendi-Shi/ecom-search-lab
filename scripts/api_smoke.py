"""Exercise the running localhost service and preserve contract results."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import urllib.error
import urllib.request

parser=argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=18866)
parser.add_argument("--output",type=Path,default=Path("outputs/api_smoke.json"))
args=parser.parse_args()
base=f"http://127.0.0.1:{args.port}"
checks=[]
with urllib.request.urlopen(base+"/health",timeout=30) as response:
    health=json.load(response)
    assert health["status"]=="ok" and health["trained_ranker"]
    checks.append({"check":"health","status":response.status,"products":health["products"],"passed":True})
with urllib.request.urlopen(base+"/",timeout=30) as response:
    page=response.read().decode()
    assert 'id="query"' in page and 'id="text-results"' in page
    checks.append({"check":"demo_page","status":response.status,"passed":True})
for payload,expected in [({"query":"running shoes without Nike","k":5,"pool":100},200),({"query":"","k":5},400),({"query":"shoes","k":True},400)]:
    request=urllib.request.Request(base+"/search",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(request,timeout=60) as response:
            status=response.status
            result=json.load(response)
    except urllib.error.HTTPError as error:
        status=error.code
        result=json.load(error)
    assert status==expected
    row={"check":"search" if expected==200 else "invalid_input","status":status,"passed":True}
    if status==200:
        assert result["ranking"]=="trained_pairwise" and result["products"]
        assert all("nike" not in p["brand"].casefold() for p in result["products"])
        row.update({"query":payload["query"],"returned":len(result["products"]),"latency_ms":result["latency_ms"],"retrieval":result["retrieval"]})
    checks.append(row)
report={"created_at_utc":datetime.now(timezone.utc).isoformat(),"base_url":base,"checks":checks,"passed":all(c["passed"] for c in checks),"scope":"single-worker local HTTP integration, not a concurrency benchmark"}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
