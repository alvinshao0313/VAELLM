"""Recover only the historical pinned official train artifacts, then verify bytes."""
import argparse
from pathlib import Path
import json
import hashlib
import time
import requests
import re

SOURCES={
 'webinstruct':dict(repository='TIGER-Lab/WebInstruct-verified',revision='3e8a350b3a935d68fe70bdf379692500abc9ff51',
     sha256='e55a8e985e18e378e4cee9059442e94ed3e8d0f6280c4b3c6e0b2231982fc975',bytes=43775058,rows=228736),
 'medmcqa':dict(repository='openlifescienceai/medmcqa',revision='91c6572c454088bf71b679ad90aa8dffcd0d5868',
     sha256='b119434ba551517a6ec0ba1f7e0b4c029165ed284a4704f262ce37c791c493c5',bytes=85899025,rows=182822)}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(root,resume=False):
    root.mkdir(parents=True,exist_ok=True);records={}
    for name,spec in SOURCES.items():
        directory=root/name;directory.mkdir(exist_ok=True)
        path=directory/'train-00000-of-00001.parquet'
        url=f"https://huggingface.co/datasets/{spec['repository']}/resolve/{spec['revision']}/data/{path.name}"
        if not path.exists():
            temporary=path.with_suffix('.partial')
            offset=temporary.stat().st_size if resume and temporary.exists() else 0
            headers={'Range':f'bytes={offset}-'} if offset else {}
            request_url=url+(f'?download=true&resume_offset={offset}' if offset else '')
            with requests.get(request_url,headers=headers,stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise ValueError('Server did not honor exact one-time Range recovery.')
                with temporary.open('ab' if offset else 'xb') as output:
                    for block in response.iter_content(1024*1024):output.write(block)
            if temporary.stat().st_size!=spec['bytes'] or sha(temporary)!=spec['sha256']:
                raise ValueError('Pinned train byte/SHA mismatch: '+name)
            temporary.rename(path)
        if path.stat().st_size!=spec['bytes'] or sha(path)!=spec['sha256']:
            raise ValueError('Retained source differs: '+str(path))
        card=directory/'README.md'
        if not card.exists():
            response=requests.get(f"https://huggingface.co/datasets/{spec['repository']}/raw/{spec['revision']}/README.md",timeout=60)
            response.raise_for_status();card.write_bytes(response.content)
        card_text=card.read_text(encoding='utf-8')
        match=re.search(r'^license:[ \t]*([^\r\n]+)',card_text,re.MULTILINE)
        if match is None:match=re.search(r'^license:[ \t]*\r?\n[ \t]*-[ \t]*([^\r\n]+)',card_text,re.MULTILINE)
        if match is None or match.group(1).strip()!='apache-2.0':raise ValueError('Pinned HF card license differs: '+name)
        records[name]=dict(spec,path=str(path),official_split='train',license_as_published=match.group(1).strip(),
            train_url=url,card_path=str(card),card_sha256=sha(card),verified_unix=time.time(),test_or_validation_downloaded=False)
        print(json.dumps(dict(event='pinned_source_ready',**records[name])),flush=True)
    (root/'sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Explicit one-time exact Range recovery; no retry loop')
    args=parser.parse_args();main(args.root,args.resume)
