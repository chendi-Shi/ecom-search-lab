"""Create verified local research deliveries; personal local profiles are excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--full',action='store_true',help='Include frozen CPU models and research serving assets for local use')
parser.add_argument('--accuracy',action='store_true',help='Include the completed accuracy upgrade in a separate full CPU archive')
args=parser.parse_args()
if args.accuracy:
    args.full=True
    for report_name in ['report.json','end_to_end_report.json']:
        report_path=ROOT/'outputs/accuracy-v1'/report_name
        if not report_path.is_file() or json.loads(report_path.read_text(encoding='utf-8')).get('status')!='completed':
            raise SystemExit('Accuracy final evaluations must complete before packaging')
destination=ROOT.parent/('ecom-search-lab-accuracy-cpu.zip' if args.accuracy else
                         'ecom-search-lab-full-cpu.zip' if args.full else 'ecom-search-lab-delivery.zip')
selected=set()

def add_file(relative):
    path=ROOT/relative
    if path.is_file():selected.add(path)

def add_directory(relative):
    path=ROOT/relative
    if path.is_dir():
        for file in path.rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix!='.pyc':selected.add(file)

for folder in ['ecom_search','tests','scripts','docs','upstream','web','data/esci','outputs/ranking']:
    add_directory(folder)
for file in ['README.md','LICENSE','NOTICE','pyproject.toml','run.ps1','run-full.ps1','.gitignore',
             'outputs/products.sqlite','outputs/products.build.json','outputs/retrieval-v4.json',
             'outputs/verification.json','outputs/verification.log','outputs/verification-stdlib.json',
             'outputs/api_smoke.json','outputs/api_smoke_full.json','outputs/experiment-summary.json',
             'outputs/ann/report.json','outputs/ann/selection.json','outputs/disk-vectors-v1/report.json',
             'outputs/disk-vectors-v1/post_measurement_neighbor_check.json',
             'outputs/disk-vectors-v1/selection.json','outputs/multimodal/report.json',
             'outputs/generative-v1/report.json','outputs/generative-v1/report.md',
             'outputs/personalization/experiment/report.json']:
    add_file(file)
if args.full:
    for folder in ['data/multimodal/images','data/multimodal/licenses',
                   'outputs/multimodal-assets/tinyclip','outputs/multimodal/selected',
                   'outputs/text-assets/minilm','outputs/disk-vectors-v1/int8']:
        add_directory(folder)
    for file in ['data/multimodal/manifest.json','data/multimodal/license_manifest.json',
                 'data/multimodal/catalog.jsonl','data/multimodal/train.jsonl',
                 'data/multimodal/val.jsonl','data/multimodal/test.jsonl']:
        add_file(file)
    # Public study artifacts only: no original review text, raw archives or test-user traces.
    for name in ['catalog.jsonl','train.jsonl','model.json','manifest.json']:
        add_file('outputs/personalization/experiment/'+name)
    for name in ['metadata.json','selection.json','decoder.pt','codebooks.npy','codes.npy',
                 'vectors.npy','ids.json','fit_product_ids.json','per_query_test.jsonl']:
        add_file('outputs/generative-v1/'+name)
    for name in ['vectors.npy','ids.json','metadata.json']:
        add_file('outputs/ann/flat/'+name)
selected=sorted(selected)
if args.accuracy:
    selected=set(selected)
    for folder in ['data/esci-v2','outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2',
                   'outputs/accuracy-v1/retrieval/int8']:
        add_directory(folder)
    for name in ['protocol.json','selection.json','model.json','validation.json','final-lock.json','validation_feature_probe.json',
                 'report.json','end_to_end_report.json','verification.json','api_smoke.json',
                 'test-bm25.jsonl','test-transferred_pairwise.jsonl','test-cross_encoder.jsonl','test-selected.jsonl']:
        add_file('outputs/accuracy-v1/'+name)
    for name in ['products.sqlite','products.build.json','pool_strategy.json','metadata.json','report.json','validation_per_query.json']:
        add_file('outputs/accuracy-v1/retrieval/'+name)
    for method in ['retrieval','transferred_pairwise','cross_encoder','selected']:
        add_file('outputs/accuracy-v1/corpus-test-'+method+'.jsonl')
    for name in ['compact-memory.json','verification.log']:
        add_file('outputs/accuracy-v1/'+name)
    for name in ['quantization_report.json','benchmark_quantization.py','int8-real-report.json']:
        add_file('outputs/cross-encoder-smoke/'+name)
    # Preserve validation-only development history, never feature matrices or caches.
    for name in ['model.json','selection.json','protocol.json','validation.json',
                 'accuracy.py','train_accuracy.py','test_accuracy.py']:
        add_file('outputs/accuracy-v1/development-initial/'+name)
    development = ROOT/'outputs/accuracy-v1/development-initial'
    if development.is_dir():
        for path in development.glob('*features*.json'):
            if path.is_file(): selected.add(path)
    add_file('outputs/accuracy-v1/development-initial/probe_accuracy_features.py')
    add_file('run-accuracy.ps1')
    add_file('requirements-accuracy.txt')
    required_accuracy = ['requirements-accuracy.txt','outputs/accuracy-v1/validation_feature_probe.json'] + [
        'outputs/accuracy-v1/development-initial/'+name for name in
        ['model.json','selection.json','protocol.json','validation.json','features-train.json','features-val.json',
         'accuracy.py','train_accuracy.py','test_accuracy.py']]
    missing=[name for name in required_accuracy if ROOT/name not in selected]
    if missing:
        raise SystemExit('Accuracy development/replication evidence is missing: '+', '.join(missing))
    selected=sorted(selected)
scope=('Local CPU research delivery with frozen text, multimodal, generative, disk-vector and Beauty serving assets; '
       'install Python and optional dependencies. Sources retain their own rights; not a grant to republish third-party data. '
       'Local user profiles, uploaded images, raw review text/archives and download caches excluded.' if args.full else
       'Portable standard-library text core, frozen ESCI data, ranker, SQLite and aggregate evidence. '
       'Optional module source included; heavy models/images/histories omitted from this small package.')
manifest={'format':3,'full':args.full,'accuracy':args.accuracy,'scope':scope,'files':[
    {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
     'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in selected]}
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in selected:archive.write(path,'ecom-search-lab/'+path.relative_to(ROOT).as_posix())
    archive.writestr('ecom-search-lab/DELIVERY_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    for row in manifest['files']:
        assert hashlib.sha256(archive.read('ecom-search-lab/'+row['path'])).hexdigest()==row['sha256']
print(json.dumps({'zip':str(destination),'bytes':destination.stat().st_size,'files':len(selected),'verified':True,'full':args.full,'accuracy':args.accuracy}))
