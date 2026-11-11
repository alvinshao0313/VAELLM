"""Bound inputs and the source9 CPU-gather bridge for the approved k/o batch."""
from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import time

from experiments.cat_auto_protocol_cmcc_20261104 import data as data_api
from experiments.cat_auto_protocol_cmcc_20261104 import live as native_live
from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha, public_consumer_identity
from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require, validate_boundary
from experiments.cat_batch_gather_cmcc_20261104 import fit as source9


APPROVAL = 'call_c5f4a5881b534505b5a2c8ce928286c1'
ORDER = ('k_proj', 'o_proj', 'gate_proj', 'q_proj', 'v_proj', 'down_proj', 'up_proj')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def bound_path(value):
    if isinstance(value, dict):
        path = Path(value['path']).resolve()
        require(file_sha(path) == value['sha256'], 'Bound input file changed: ' + str(path))
        return path
    return Path(value).resolve()


def source_manifest(request):
    return read(bound_path(request['source_manifest']))


def load_inputs(request, *, with_sources=True):
    """Reuse the actual DataPlan; source loading is needed only for recovery."""
    import torch
    from e2e_common.data import build_tokenizer

    inventory_path = bound_path(request['inventory_receipt'])
    data_path = bound_path(request['data_plan_path'])
    if request.get('prepared_data_path'):
        require(bound_path(request['prepared_data_path']) == data_path, 'Two different prepared DataPlans were supplied.')
    inventory = read(inventory_path)['inventory']
    prepared = torch.load(data_path, map_location='cpu', weights_only=False)
    require(digest(prepared['manifest']) == prepared['data_identity']
        and prepared['manifest']['model_identity'] == inventory['model_identity'],
        'The prepared data manifest does not belong to the actual model inventory.')
    if request.get('source_manifest'):
        manifest = source_manifest(request)
        require(inventory_path == bound_path(manifest['inventory']) and data_path == bound_path(manifest['data_plan'])
            and prepared['data_identity'] == manifest['run_identity']['data_identity']
            and prepared['manifest']['search_seed'] == manifest['run_identity']['search_seed'],
            'The live stage changed the frozen source9 inventory, DataPlan or seed.')
    tokenizer = build_tokenizer(inventory['source_evidence']['path'], None)
    require(data_api.tokenizer_manifest(tokenizer) == prepared['manifest']['tokenizer'], 'Actual tokenizer differs from the frozen DataPlan.')
    sources = []
    if with_sources:
        registry = prepared['local_sources_registry']
        require(tuple(row['alias'] for row in registry['sources']) == tuple(data_api.MIX), 'Keep the original five train sources and enumeration.')
        for row in registry['sources']:
            source = data_api.load_local_source(row['alias'], revision='converted-artifact-sha256:' + row['sha256'],
                license_record=row['license'], provenance_record=row['provenance'], expected_sha256=row['sha256'])
            require(len(source.raw) == row['rows'], 'Actual train source row count changed.')
            sources.append(source)
        require([source.identity for source in sources] == prepared['manifest']['sources'], 'Actual source provenance/fingerprints changed from the frozen stream.')
    return inventory, prepared, sources, tokenizer


def verify_fit_sources(manifest):
    """Verify semantic fitter/consumer pins without changing old orchestration."""
    require(source9.fit_code_identity() == manifest['run_identity']['code_identity'], 'The actual source9 fitter or CPU gather changed.')
    require(public_consumer_identity() == manifest['run_identity']['consumer_identity'], 'The actual PUBLIC consumer changed.')
    native_live.source_pins()
    return dict(fit_code_identity=source9.fit_code_identity(), consumer_identity=public_consumer_identity(),
        live_source_pins=native_live.source_pins(), implementation=binding(__file__))


def actual_parent(request):
    parent = request['parent']
    if isinstance(parent, str) or isinstance(parent, dict) and 'sha256' in parent and 'path' in parent:
        parent = read(bound_path(parent))
    return validate_boundary(parent)


def live_identity(manifest, parent):
    identity = deepcopy(manifest['run_identity'])
    identity.update(parent_uuid=parent['uuid'], parent_weights_digest=parent['digest'],
        code_identity=source9.fit_code_identity(), consumer_identity=public_consumer_identity())
    return identity


def validate_execution(request):
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae', 'Activate the confirmed bitvae environment in the caller shell.')
    require(request['phase'] in ('formal', 'technical_validation'), 'Use an actual formal or technical live stage.')
    require(request['category_id'] in ORDER[:2] and request['recipe']['category_id'] == request['category_id'], 'This batch only fits k_proj/o_proj.')
    require(type(request['gpu']) is int and request['gpu'] in range(6)
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(request['gpu']), 'Use exactly the physical GPU assigned by the batch controller.')
    require(os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True', 'Keep the verified allocator policy.')
    require(time.time() < request['deadline_epoch'], 'The controller-owned finite stage deadline expired.')
    authorization = read(bound_path(request['authorization']))
    require(authorization.get('confirmed') is True and authorization.get('user_approved') is True
        and authorization.get('authorization_reference') == APPROVAL
        and authorization.get('total_GPU_hours') == 36 and authorization.get('gpus') == list(range(6))
        and authorization.get('budget_by_component') == dict(technical=4, fit=6, recovery=16, evaluation=9, boundary=1),
        'Bind the exact confirmed36GPUh k/o batch and its component limits.')
    require(Path(request['output']).resolve().is_relative_to(Path(authorization['output_root']).resolve()),
        'The worker output must stay inside this authorized isolated batch root.')
    return authorization


def materialize_root(request, destination):
    """CPU-only real BF16→v6 root; both k branches must share this reference."""
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Materialize the shared dense root on CPU with CUDA hidden.')
    inventory, _prepared, _sources, tokenizer = load_inputs(request, with_sources=False)
    manifest = source_manifest(request)
    verify_fit_sources(manifest)
    source = Path(inventory['source_evidence']['path'])
    require(all(file_sha(source / name) == item['sha256'] for name, item in inventory['source_evidence']['files'].items()), 'The verified BF16 source bytes changed.')
    # No CPU forward occurs. The actual fit/recovery factory selects its own
    # explicit CUDA attention policy when reconstructing this exact payload.
    return native_live.materialize_bf16_root(inventory=inventory, frozen_order=ORDER,
        destination=destination, attention_implementation='sdpa', tokenizer=tokenizer, phase='formal')


class _Source9Bridge(ast.NodeTransformer):
    def __init__(self):
        self.fit_imports = self.returns = 0

    def visit_ImportFrom(self, node):
        if node.level == 1 and node.module == 'fit_adapter' and [item.name for item in node.names] == ['fit_category']:
            self.fit_imports += 1
            node.level, node.module = 0, 'experiments.cat_batch_gather_cmcc_20261104.fit'
        return node

    def visit_Return(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'dict':
            self.returns += 1
            node.value.keywords.extend([
                ast.keyword(arg='_fitted_model', value=ast.Name(id='model', ctx=ast.Load())),
                ast.keyword(arg='_category_bank', value=ast.Subscript(value=ast.Name(id='fitted', ctx=ast.Load()),
                    slice=ast.Constant(value='category_bank'), ctx=ast.Load()))])
        return node


def source9_live_entry():
    """Compile the private bridge without mutating the original module globals."""
    tree = ast.parse(inspect.getsource(native_live.fit_next_live_category))
    adapter = _Source9Bridge()
    tree = ast.fix_missing_locations(adapter.visit(tree))
    require((adapter.fit_imports, adapter.returns) == (1, 1), 'The pinned native live adapter changed its import/return structure.')
    namespace = dict(native_live.__dict__)
    exec(compile(tree, native_live.__file__, 'exec'), namespace)
    return namespace['fit_next_live_category']


def source9_live_fit(**kwargs):
    """Keep the real live adapter; replace its one local fit import, privately."""
    return source9_live_entry()(**kwargs)
