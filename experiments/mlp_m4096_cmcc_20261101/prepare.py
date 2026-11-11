"""Build the same official hash prefix; compare complete1024 prefix before caching."""
from pathlib import Path
import time

from datasets import Dataset

from . import bindings as b
from . import data, pool_build
from .fetch import SOURCE,REVISION


def main():
    b.executor.cpu_environment()
    b.torch.set_num_threads(4)
    b.require(not b.DATA.exists(),'Use a new isolated coverage data directory.')
    capture=b.read(SOURCE/'capture.json')
    b.require(capture['revision']==REVISION and capture['repository']=='cais/mmlu'
        and capture['config']=='auxiliary_train' and capture['split']=='train','Fixed official source changed.')
    for record in capture['files'].values(): data.checked(record)
    auxiliary=Dataset.from_parquet([str(SOURCE/name) for name in capture['selected_parquet']],cache_dir=str(SOURCE/'arrow_cache'))
    b.require(len(auxiliary)==99842 and auxiliary.column_names==['train'],'Fixed auxiliary row layout/count changed.')
    old,recovery,evidence=data.control.load()
    source=b.read(data.original.HOT.parent/'qwen_teacher_scope_qa128_20261011/dataset_prepared.json')['data_contract']['mmlu']['source']
    tokenizer=pool_build.original.helpers.core.build_tokenizer(str(b.HF))
    m_cases,selection,template=pool_build.build(tokenizer,auxiliary,source)
    verifier=data.legacy.module(data.original.HOT/'common.py','_M4096_prefix_verifier')
    verifier.require_same(m_cases[:1024],old['training_cases'][:1024])
    # Existing prefix is reused directly; its historical path remains an immutable coordinate.
    # New cases point at this new official capture, and never claim the absent old Arrow exists.
    for case in m_cases[1024:]:
        case['source']['data_path']=str(SOURCE)
    cases=old['training_cases'][:1024]+m_cases[1024:]+old['training_cases'][1024:]
    dataset=dict(training_cases=cases,groups=data.groups(cases,old),train_ids=[case['case_id'] for case in cases],
        evidence=dict(source_capture=b.binding(SOURCE/'capture.json'),logical_original_source=source,
            selected_auxiliary_rows=selection,native_template=template,original_prefix_exact=True))
    properties=data.validate(dataset,old)
    b.DATA.mkdir(parents=True)
    b.torch.save(dataset,b.DATA/'dataset.pt')
    saved=b.torch.load(b.DATA/'dataset.pt',map_location='cpu',weights_only=True)
    verifier.require_same(dataset,saved)
    shapes=data.input_shapes(dataset)
    source_pins={str(path.resolve()):b.sha(path) for path in (b.HERE/'prepare.py',b.HERE/'pool_build.py',b.HERE/'data.py',
        Path(pool_build.original.__file__),Path(pool_build.original.native.__file__),Path(pool_build.original.data.__file__))}
    record=dict(schema='same_revision_M4096_new_capture_prepared_v1',status='actual_CPU_prefix4096_complete',
        dataset=b.binding(b.DATA/'dataset.pt'),properties=properties,source_capture=b.binding(SOURCE/'capture.json'),
        source_sha256=source_pins,control_data=evidence,source=source,selection=selection,
        groups=shapes,complete_first1024_values_exact=True,old_Arrow_available=False,new_capture=True,
        historical_old_path_only_in_reused_prefix=True,prepared_unix=time.time(),CUDA_initialized=b.torch.cuda.is_initialized())
    b.write(b.DATA/'dataset_prepared.json',record)
    print(dict(status=record['status'],properties=properties,peak=max(shapes,key=lambda row:row['M_max_length'])),flush=True)


if __name__=='__main__':main()
