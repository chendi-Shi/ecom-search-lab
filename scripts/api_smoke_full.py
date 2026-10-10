"""Real localhost integration checks; runs against frozen experiment artifacts."""
import argparse
import atexit
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import quote
import uuid
import struct
import zlib

parser=argparse.ArgumentParser()
parser.add_argument('--port',type=int,default=18866)
parser.add_argument('--output',type=Path,default=Path('outputs/api_smoke_full.json'))
args=parser.parse_args()
base=f'http://127.0.0.1:{args.port}'
checks=[]
completed=False

def save_status(status):
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'base_url':base,'checks':checks,'passed':status=='completed','status':status,'scope':'Actual single-worker CPU HTTP integration; latency includes encoding, is not a load test.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

save_status('running')
atexit.register(lambda: None if completed else save_status('failed'))

def request(path,obj=None,expected=200,binary=False):
    req=urllib.request.Request(base+path,data=None if obj is None else json.dumps(obj).encode(),headers={} if obj is None else {'Content-Type':'application/json'})
    try:
        response=urllib.request.urlopen(req,timeout=180)
    except urllib.error.HTTPError as error:response=error
    with response:
        assert response.status==expected,(path,response.status,response.read())
        return response.read() if binary else json.load(response)

def passed(name,**details):checks.append({'check':name,'passed':True,**details})

health=request('/health')
assert health['status']=='ok' and all(health[name] for name in ['trained_ranker','generative','multimodal','beauty','disk'])
passed('all_modules_loaded',products=health['products'])
page=request('/',binary=True).decode()
assert all(f'id="{value}"' in page for value in ['text-results','image-results','beauty-results','report-results'])
passed('four_panel_page')
user='integration-'+uuid.uuid4().hex
try:
    result=request('/search',{'query':'running shoes without Nike','k':8,'pool':100,'user_id':user,'use_generative':False})
    assert result['products'] and all('nike' not in (p['title']+' '+p['brand']).lower() for p in result['products'])
    passed('disk_hybrid_negative_constraint',returned=len(result['products']),latency_ms=result['latency_ms'])
    pid=result['products'][0]['id']
    request('/profile',{'user_id':user,'product_id':pid,'action':'like'})
    personal=request('/search',{'query':'running shoes without Nike','k':8,'pool':100,'user_id':user,'use_generative':False})
    assert personal['personalization']['applied'] and any('preference_score' in p for p in personal['products'])
    assert any(p['id']==pid and p['preference_score']>0 for p in personal['products'])
    passed('explicit_preference_applied')
    request('/profile',{'user_id':user,'product_id':pid,'action':'hide'})
    hidden=request('/search',{'query':'running shoes without Nike','k':8,'pool':100,'user_id':user,'use_generative':False})
    assert all(p['id']!=pid for p in hidden['products'])
    passed('explicit_hide_filters_result')
    gen=request('/search',{'query':'wireless headphones','k':5,'pool':100,'use_generative':True})
    assert 'generative_sid' in gen['retrieval'] and gen['products']
    passed('autoregressive_retrieval_integrated',returned=len(gen['products']),latency_ms=gen['latency_ms'])
finally:
    request('/profile',{'user_id':user,'action':'clear'})

catalog=request('/multimodal/catalog')['products']
assert catalog
pid=catalog[0]['id']
raw=request('/image/'+quote(pid,safe=''),binary=True)
assert len(raw)>100
passed('real_catalog_image_served',bytes=len(raw))
image=request('/multimodal/search',{'reference_id':pid,'query':'black dress','mode':'learned','k':5})
assert len(image['products'])==5 and all('score' in p for p in image['products'])
passed('reference_image_and_text_search',latency_ms=image['latency_ms'])
upload=request('/multimodal/search',{'image_base64':base64.b64encode(raw).decode(),'mode':'image','k':5})
assert len(upload['products'])==5
passed('uploaded_real_image_search',latency_ms=upload['latency_ms'])
text=request('/multimodal/search',{'query':'black dress','image_base64':'stale invalid upload','mode':'text','k':5})
assert len(text['products'])==5
passed('text_mode_ignores_unused_image')
users=request('/recommend/users')['users']
assert users
recommend=request('/recommend?user='+quote(users[0],safe=''))
assert recommend['products'] and recommend['products'][0]['personal_weight']==.25
passed('frozen_user_history_recommendation',returned=len(recommend['products']))
assert request('/experiments')['cards']
passed('experiment_results_available')
for name,path,payload in [
    ('reject_bool_k','/search',{'query':'shoes','k':True}),
    ('reject_nonbool_mode','/search',{'query':'shoes','use_generative':'false'}),
    ('reject_unknown_profile_item','/profile',{'user_id':user,'product_id':'unknown','action':'like'}),
    ('reject_arbitrary_image_path','/multimodal/search',{'image_path':'C:/Windows/win.ini'}),
    ('reject_invalid_image','/multimodal/search',{'image_base64':base64.b64encode(b'not an image').decode()}),
]:
    assert request(path,payload,expected=400)['error']
    passed(name)
def png_chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
huge=b'\x89PNG\r\n\x1a\n'+png_chunk(b'IHDR',struct.pack('>IIBBBBB',50000,50000,8,2,0,0,0))+png_chunk(b'IDAT',zlib.compress(b'\0'*4))+png_chunk(b'IEND',b'')
assert request('/multimodal/search',{'image_base64':base64.b64encode(huge).decode(),'mode':'image'},expected=400)['error']
passed('reject_decompression_bomb_as_json')
completed=True
report=save_status('completed')
print(json.dumps(report))
