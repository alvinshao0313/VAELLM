"""CPU native inputs, source signatures, exact QA matching, and peak group selection."""
import argparse
import copy
from collections import Counter
from pathlib import Path
import shutil
import os
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from . import bindings as b,data


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preparation required.')
    b.require(not b.DATA.exists(),'Use new isolated data path; no overwrite.')
    med,web=data.helpers();sources=b.read(b.RAW/'sources.json')
    for source in sources.values():
        b.require(b.sha(source['path'])==source['sha256'] and Path(source['path']).stat().st_size==source['bytes']
            and b.sha(source['card_path'])==source['card_sha256'] and source['official_split']=='train'
            and source['license_as_published']=='apache-2.0' and not source['test_or_validation_downloaded'],
            'Pinned official train bytes/card/split/license changed.')
    old,prior,evidence=data.original.load()
    tokenizer=AutoTokenizer.from_pretrained(str(b.HF),use_fast=True,local_files_only=True)
    codec=med.native.PairTokenizer(tokenizer);task,_=med.native_task()
    by_source={};heldout={};selections={};counts={}
    for name in ['webinstruct','medmcqa']:
        spec=sources[name];rows=pq.read_table(spec['path'],use_threads=False).to_pylist()
        b.require(len(rows)==spec['rows'],'Official train row count changed.')
        qualified=[]
        for index,row in enumerate(rows):
            if name=='webinstruct':
                if row['answer_type']!='Multiple Choice':continue
                reason,parsed,_=web.inventory.inspect_row(row)
                if reason!='qualified':continue
            else:parsed=None
            key=('webinstruct' if name=='webinstruct' else 'medmcqa')+'/train/'+row['id']
            qualified.append((med.rank(key),key,index,row,parsed))
        b.require(len(qualified)==(11219 if name=='webinstruct' else 182822)
            and len({row[1] for row in qualified})==len(qualified),'Original structural supply/author IDs changed.')
        ranked=sorted(qualified);selected=ranked[:2048];diagnostic=ranked[2048:2112]
        cases=[];extra=[]
        for position,(_,key,index,row,parsed) in enumerate(selected+diagnostic):
            if name=='webinstruct':
                source=dict(spec,dataset=spec['repository'],train_path=spec['path'])
                case=web.build_case(row,index,parsed,codec,source)
            else:
                source=dict(spec,local_path=spec['path'])
                case=med.build_med_case(row,index,task,codec,dict(source=source))
            b.require(len(case['input_ids'])+1<=32768,'Real selected input exceeds supported length; do not skip.')
            (cases if position<2048 else extra).append(case)
        by_source[name]=cases;heldout[name]=extra
        selections[name]=[dict(rank=i,hash=entry[0],case_id=entry[1],row=entry[2],input_signature=cases[i]['input_signature'])
            for i,entry in enumerate(selected)]
        counts[name]=dict(official_rows=len(rows),structurally_eligible=len(qualified),selected=2048,
            shortest=min(len(case['input_ids']) for case in cases),longest=max(len(case['input_ids']) for case in cases),
            tokens=sum(len(case['input_ids']) for case in cases),heldout64_max=max(len(case['input_ids']) for case in extra))
    groups=[];ordered=[]
    for index,previous in enumerate(old['groups']):
        batch=by_source['webinstruct'][index*16:(index+1)*16]+by_source['medmcqa'][index*16:(index+1)*16]
        ordered.extend(batch);ids=[case['case_id'] for case in batch]+previous['case_ids'][32:]
        groups.append(dict(previous,case_ids=ids,input_signatures=[case['input_signature'] for case in batch]+previous['input_signatures'][32:],
            weights={key:.5/(32 if slot<32 else 8) for slot,key in enumerate(ids)}))
        b.require({k:v for k,v in groups[-1].items() if k not in ('case_ids','input_signatures','weights')}
            =={k:v for k,v in previous.items() if k not in ('case_ids','input_signatures','weights')},'Original group metadata changed.')
    qa=old['training_cases'][4096:];med.require_same(qa,old['training_cases'][4096:])
    dataset=dict(old,training_cases=ordered+qa,train_ids=[case['case_id'] for case in ordered+qa],groups=groups,
        evidence=dict(schema='S4_WebMed_native4_original_QA_v1',M_sources=sources,QA_original=True))
    for group in groups:
        cases=ordered[group['group_index']*32:(group['group_index']+1)*32]
        packed,options=data.legacy.mmlu.batch_inputs(cases,tokenizer.pad_token_id)
        b.require(packed['input_ids'].shape[0]==32 and options.tolist()==[[362,425,356,422]]*32,
            'Actual unchanged M32 packer or native option support changed.')
    shapes=data.input_shapes(dataset)
    peak_M=max(shapes,key=lambda row:row['M_max_length'])['group_index']
    peak_QA=max(shapes,key=lambda row:row['QA_max_length'])['group_index']
    # The same parent already covered M32x1222 and the unchanged QA10593.
    short_indices=[peak_M,(peak_M+1)%128] if shapes[peak_M]['M_max_length']>1222 else []
    held=[*heldout['webinstruct'],*heldout['medmcqa']]
    b.require(not {case['case_id'] for case in ordered}&{case['case_id'] for case in held},'Heldout entered train.')
    b.DATA.mkdir(parents=True)
    b.torch.save(dataset,b.DATA/'dataset.pt');b.torch.save(held,b.DATA/'heldout.pt')
    med.require_same(dataset,b.torch.load(b.DATA/'dataset.pt',map_location='cpu',weights_only=True))
    shutil.copyfile(b.UPSTREAM/'probe_inputs.pt',b.RUN/'fixture.pt')
    record=dict(dataset=b.binding(b.DATA/'dataset.pt'),heldout=b.binding(b.DATA/'heldout.pt'),sources=sources,
        selection_rule='SHA256(causal_output_fit/137/+source/train/<authorID>), tie case_id; first2048; no length/teacher/test selection',
        selection=selections,summary=counts,shapes=shapes,short_indices=short_indices,QA_exact=True,
        inherited_QA_source_digest=evidence['actual_input_target_records_sha256'],CUDA_initialized=b.torch.cuda.is_initialized())
    b.require(not record['CUDA_initialized'],'CPU prepare initialized CUDA.')
    b.write(b.DATA/'dataset_prepared.json',record)
    print(dict(event='CPU_native_WebMed_inputs_verified',summary=counts,short_indices=short_indices,
        peak_M=shapes[peak_M],peak_QA=shapes[peak_QA],heldout128=True,original_QA_exact=True),flush=True)


if __name__=='__main__':main()
