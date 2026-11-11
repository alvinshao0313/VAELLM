"""A real BR36 endpoint feeding the unchanged H96/Muon32 mathematics."""
from copy import deepcopy
from pathlib import Path

# This read-only adapter closes the already selected N namespace before legacy
# data helpers load. Its source reader is deliberately not used for BR parents.
from experiments.fresh_muon_cmcc_20261030 import bindings as direct
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO

mature, fit, inputs = direct.mature, direct.fit, direct.inputs
lifecycle, optimizer = direct.lifecycle, direct.optimizer
torch, core, checkpoint = direct.torch, direct.core, direct.checkpoint
read, require, binding = direct.read, direct.require, direct.binding
shared, SUPPORT, HF = direct.shared, direct.SUPPORT, direct.HF
HERE = Path(__file__).resolve().parent
GROUPS = list(range(32))
RUN_NAME = 'BR_E32'
CONTRACT = {**deepcopy(mature.CONTRACT),
    'student_source': 'Verified complete C0 -> BR36 with32 updates per block -> independent fresh BR endpoint; actual BR UUID.',
    'budget': 'H96 M1024 once/Q32 twice/H96 twice; original500 linear prefix32; seed137/R0; complete full9 decides quality.',
    'initialization': 'BR BF16 decoders -> fresh FP32 masters/optimizer; recompute first-gradient kappa and all student QA/HF trajectories.',
    'comparison': 'Same mathematics, document presentations and32 updates as direct C0->E; actual online supervision is reported separately.',
    'lineage': 'Preserve the actual ancestor C0 binding and the complete BR record; never relabel BR as C0.'}


def configure():
    from . import bindings as formal_BR
    formal_BR.configure_readers()
    mature.STEPS = {'formal32': GROUPS}


def source(checkpoint_dir, report_path):
    checkpoint_dir, report_path = Path(checkpoint_dir).resolve(), Path(report_path).resolve()
    require(report_path.is_file(), 'The real complete BR36 report is missing; no future, native or short source is eligible.')
    report = read(report_path)
    require(report.get('schema') == 'BR36_H96_32_execution_v1' and report.get('validation_only') is False,
            'BR->E requires the formal complete36-block/32-update BR record, never the adjacent-block prototype.')
    from .source import source as formal_BR_source
    parent, meta = formal_BR_source(checkpoint_dir, report_path)
    require(parent['checkpoint_id'] != parent['source_C0_binding']['checkpoint_id']
        and parent['BR_record'], 'Keep the genuine BR UUID, complete BR record and distinct ancestor C0 binding.')
    return parent, meta


def verified_implementation():
    return direct.verified_implementation()


def code_bindings():
    from . import bindings as formal_BR
    paths = set(HERE.glob('*.py'))
    paths.update(mature.HERE/name for name in ('bindings.py', 'fit.py', 'inputs.py', 'lifecycle.py',
                                               'optimizer.py', 'rollout.py', 'motion.py', 'prepare.py'))
    paths.update((Path(direct.__file__), direct.HERE/'prepare.py', direct.HERE/'run.py',
                  Path(shared.__file__), SUPPORT/'prepared.json'))
    return {**{path:binding(path) for path in formal_BR.code_bindings()},
            **{str(path.resolve()):binding(path) for path in paths}}


def read_prepared(path):
    value = read(Path(path).resolve())
    require(value['schema'] == 'BR_E32_prepared_v1'
        and value['status'] == 'verified_actual_BR36_source_and_reused_Muon_GPU_contract'
        and value['contract'] == CONTRACT and value['group_indices'] == GROUPS,
        'Use this explicit genuine-BR H96/Muon32 preparation.')
    for item in value['bindings'].values():
        require(binding(item['path']) == item, 'Prepared BR->E code, source or data changed: '+item['path'])
    proof = verified_implementation()
    require(value['GPU_evidence'] == {key:item for key,item in proof.items() if key != 'record'},
        'The reused actual mixed Muon GPU proof changed after BR->E preparation.')
    for item in value['record']['teacher_identity']['hf_shards']:
        require(binding(item['path']) == item, 'Prepared original full-HF teacher shard changed: '+item['path'])
    for path, digest in value['record']['teacher_identity']['hf_metadata_sha256'].items():
        require(mature.sha256(path) == digest, 'Prepared HF/tokenizer metadata changed: '+path)
    parent, _meta = source(value['source_binding']['checkpoint_dir'], value['source_binding']['pipeline_report'])
    require(parent == value['source_binding']
        and value['record']['source_checkpoint_id'] == parent['checkpoint_id']
        and value['source_C0_binding'] == parent['source_C0_binding']
        and value['BR_record'] == parent['BR_record'], 'The prepared actual BR endpoint or ancestor lineage changed.')
    return value
