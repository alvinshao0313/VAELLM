"""Recover only the fixed official auxiliary_train source missing after migration."""
import os
import hashlib
from pathlib import Path
import traceback

from huggingface_hub import HfApi
import requests

from . import bindings as b

REVISION = 'c30699e8356da336a370243923dbaf21066bb9fe'
SOURCE = Path('/data1/cmcc/VAELLM/data/cais_mmlu_auxiliary_train_'+REVISION)


def main():
    b.executor.cpu_environment()
    SOURCE.mkdir(parents=True,exist_ok=True)
    b.require(not (SOURCE/'capture.json').exists(), 'Do not overwrite an official-source capture.')
    api = HfApi()
    info=api.dataset_info('cais/mmlu',revision=REVISION,files_metadata=True)
    files = [row.rfilename for row in info.siblings]
    metadata={row.rfilename:row for row in info.siblings}
    selected = sorted(path for path in files if path.startswith('auxiliary_train/') and path.endswith('.parquet'))
    b.require(selected, 'The fixed official revision has no auxiliary_train Parquet source.')
    fetched = {}
    for name in ['README.md',*selected]:
        path = SOURCE/name
        row=metadata[name]
        expected=row.lfs.sha256 if row.lfs else None
        path.parent.mkdir(parents=True,exist_ok=True)
        url='https://hf-mirror.com/datasets/cais/mmlu/resolve/'+REVISION+'/'+name
        if not path.exists():
            with requests.Session() as session:
                with session.get(url,stream=True,timeout=(15,120)) as response:
                    response.raise_for_status()
                    with path.open('xb') as stream:
                        for chunk in response.iter_content(1024*1024): stream.write(chunk)
        b.require(path.stat().st_size==row.size and (b.sha(path)==expected if expected else
            hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0'+path.read_bytes()).hexdigest()==row.blob_id),
            'Official pinned file size/digest differs: '+name)
        print({'fetched':name,'bytes':path.stat().st_size},flush=True)
        fetched[name] = b.binding(path)
    b.write(SOURCE/'capture.json',dict(schema='same_revision_official_auxiliary_train_capture_v1',
        repository='cais/mmlu',revision=REVISION,config='auxiliary_train',split='train',
        source_URL='https://huggingface.co/datasets/cais/mmlu/tree/'+REVISION,
        transport='https://hf-mirror.com; each file verified against the pinned repository size and LFS SHA256/git blob',
        files=fetched,selected_parquet=selected,license='MIT; fixed-revision README.md preserved',
        old_Arrow_missing=True,old_Arrow_SHA256='af0bbd09f5648bf5841df34708d700065fa7a3eb973953c1c4a1817e52af6bf0',
        new_capture_not_old_Arrow=True))
    print({'status':'same_revision_source_recovered','files':list(fetched)},flush=True)


if __name__=='__main__':
    main()
