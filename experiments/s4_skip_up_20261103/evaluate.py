"""Keep the proven consumer; distinguish local fixtures from actual HF-origin runs."""
from pathlib import Path
from . import bindings as b

_core=b.bound_implementation('evaluate')


def load(directory):
    directory=Path(directory).resolve(strict=True)
    b.require(directory.is_relative_to(b.OUTPUT),'Only this isolated candidate may be consumed.')
    model,meta,loaded=_core.load_v6_model_checkpoint(str(directory),base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys,'New candidate native strict load failed.')
    provenance=meta['extra_meta']['distill_stage']
    b.require(provenance['algorithm']=='fresh_BF16_S4_prefix_recovery'
        and provenance['contract']['candidate']==b.CONTRACT['candidate'],'Candidate/source contract mismatch.')
    if provenance.get('validation_only'):
        b.require(provenance['origin']=='incremental_validation_fixture' and provenance['incremental_fixture']['validation_only']
            and 'validation' in directory.parts,'A local fixture must never be labeled a formal HF-origin candidate.')
    else:b.require(provenance['origin']=='original_HF_BF16' and not provenance.get('incremental_fixture'),
        'Formal candidate must come from the new HF algorithm, never a validation fixture.')
    expected=b.read(directory.parent/'expected_state.json')
    b.require(_core.runtime.records(model.state_dict())==expected,'Actual exported tensor state differs.')
    budget=_core.audit_state_dict(meta,model.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'],'Actual storage contract failed.')
    if len(meta['converted_modules'])==252:
        b.require(abs(budget['user_linear_bpw']-4.094031292459239)<1e-12,'FullS4 actual ledger changed.')
    model.eval().requires_grad_(False)
    return model,meta,budget


_core.load=load

if __name__=='__main__':_core.main()
