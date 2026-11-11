"""New-run input/authorization checks; no old batch authorization is inherited."""
import os
from pathlib import Path
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.sc_cat_live_cmcc_20261106.bindings import read, write, binding, bound_path
from experiments.sc_cat_live_cmcc_20261106.live_io import load_inputs
from .identity import source_pins


def validate_request(request, *, cpu_controller=False):
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae', 'Activate bitvae in the caller shell.')
    manifest = read(bound_path(request['manifest']))
    require(request['category_id'] in manifest['frozen_order'] and request['phase'] in ('formal','technical_validation'), 'Invalid category or phase.')
    auth = read(bound_path(request['authorization']))
    require(auth.get('confirmed') is True and auth.get('user_approved') is True and auth.get('authorization_reference')
        and auth['run_id']==manifest['run_id'] and auth['manifest']==request['manifest']
        and request['category_id'] in auth['categories'] and auth['gpus']==[4,5,6,7], 'Explicit new-run GPU budget approval is required.')
    require(request['gpu'] in auth['gpus'] and time.time()<request['deadline_epoch'], 'GPU assignment or finite deadline invalid.')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==('' if cpu_controller else str(request['gpu']))
        and os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True', 'Use exactly the assigned device and confirmed allocator.')
    for key in ('output','destination'):
        if request.get(key):
            require(Path(request[key]).resolve().is_relative_to(Path(manifest['output_root']).resolve()), 'Output escapes the isolated new run.')
    require(source_pins()==manifest['recovery_source_sha256'], 'Approved recovery implementation changed; revalidate affected paths.')
    for key in ('inventory_receipt','data_plan_path'):
        require(request[key]==manifest[{'inventory_receipt':'inventory','data_plan_path':'data_plan'}[key]], 'Changed model/data input.')
    return manifest, auth
