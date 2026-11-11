"""Real fresh Rice lineage feeding the unchanged mature mixed Muon mathematics."""
from copy import deepcopy
from pathlib import Path

# C1 selects and closes N before any legacy objective/data helper imports.
from experiments.fresh_rice_cmcc_20261031 import evaluate as c1
from experiments.fresh_muon_cmcc_20261030 import bindings as direct
from experiments.cmcc_runtime_20261030 import sequence as executor
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO, TOPIC

mature, fit, inputs = direct.mature, direct.fit, direct.inputs
lifecycle, optimizer = direct.lifecycle, direct.optimizer
torch, core, checkpoint = direct.torch, direct.core, direct.checkpoint
read, require, binding = direct.read, direct.require, direct.binding
SUPPORT, HF = direct.SUPPORT, direct.HF
HERE = Path(__file__).resolve().parent
OUTPUT = ARTIFACTS / 'rice_muon_cmcc_20261031'
EVIDENCE = TOPIC / 'rice_muon_cmcc_20261031'
CONTROL_PLAN = TOPIC / 'fresh_rice_cmcc_20261031/single_20261031/plan.json'
SOURCE = ARTIFACTS / 'fresh_rice_cmcc_20261031/formal'
GROUPS = list(range(32))
PHASES = dict(short2=[0, 1], formal32=GROUPS)
CONTRACT = {**deepcopy(mature.CONTRACT),
    'student_source': 'The completed fresh C1_Rice final UUID from its own common after_o/raw2; no C0 or old Rice substitution.',
    'budget': 'H96 M1024 once/Q32 twice/H96 twice; original500 linear prefix32; seed137/R0; separate engineering group0/1 lifecycle2.',
    'initialization': 'Original C1 BF16 D -> fresh FP32 masters/optimizer for each run; recompute kappa and every student/QA teacher trajectory.',
    'lineage': 'Engineering2 and formal32 load the same original C1 final independently; never continue short2 with30 updates.',
    'frozen': 'All non-D persistent fields, including every Rice index/value/scale/zero/header, codes, protection and rotation/signs.'}


def configure():
    # The idempotent reader never replaces an installed Rice-Muon dispatch.
    c1.configure_readers(SUPPORT)
    mature.STEPS = deepcopy(PHASES)


def source(checkpoint_dir, report_path):
    configure()
    checkpoint_dir, report_path = Path(checkpoint_dir).resolve(), Path(report_path).resolve()
    require(checkpoint_dir == SOURCE / 'final_model' and report_path == SOURCE / 'report.json',
        'Use only this fresh C1 final and report, never a C0, old Rice, short or intermediate source.')
    # C1 checks its completion header before inspecting weights; pending parents fail.
    return c1.source(checkpoint_dir, report_path)


def verified_implementation():
    return direct.verified_implementation()


def code_bindings():
    paths = set(HERE.glob('*.py'))
    paths.update(Path(path) for path in c1.s.source_pins())
    paths.update(mature.HERE / name for name in ('bindings.py', 'fit.py', 'inputs.py', 'lifecycle.py',
        'optimizer.py', 'rollout.py', 'motion.py', 'prepare.py'))
    paths.update((Path(direct.__file__), direct.HERE / 'prepare.py', direct.HERE / 'run.py',
        direct.HERE / 'after_control.py', Path(executor.__file__), Path(c1.cmcc.__file__), SUPPORT / 'prepared.json'))
    return {str(path.resolve()): binding(path) for path in paths}


def rice_proof(model):
    state = model.state_dict()
    names = {f'model.layers.{layer}.mlp.gate_proj.sparse_{field}'
        for layer in range(36) for field in c1.s.sparse.FIELDS}
    actual = {name: core.tensor_record(value) for name, value in state.items() if '.mlp.gate_proj.sparse_' in name}
    require(set(actual) == names, 'Every actual Rice gate must retain its complete eight persistent fields.')
    for layer in range(36):
        name = f'model.layers.{layer}.mlp.gate_proj'
        module = model.get_submodule(name)
        require(isinstance(module, c1.s.sparse.Fixed2RiceLinear) and module.metadata()['implicit_length'] == 2
            and bool((module.lengths == 2).all()) and name + '.length_frame' not in state
            and not any(key.startswith(name + '.sparse_decoded_') for key in state),
            'Use actual implicit length2; decoded row/col/value caches must not become persistent deployment fields.')
    return dict(gates=36, tensors=len(actual), sha256=core.mapping_digest(actual),
        fields_per_gate=list(c1.s.sparse.FIELDS), implicit_length2=True, persisted_route_frame=False,
        persisted_decoded_sparse_cache=False, scope='All actual Rice indices, values, scales, zeros and stream metadata.')


def read_prepared(path):
    path = Path(path).resolve()
    value = read(path)
    require(value['schema'] == 'fresh_C1_Rice_Muon_prepared_v1'
        and value['status'] == 'verified_actual_fresh_Rice_source_Muon_CPU_prepared'
        and value['contract'] == CONTRACT and value['group_indices'] == GROUPS,
        'Use the explicit real-C1 preparation, never a short deployment or old C0 schema.')
    for item in value['bindings'].values():
        require(binding(item['path']) == item, 'Prepared source/input changed: ' + item['path'])
    proof = verified_implementation()
    require(value['reused_fixed2_GPU_evidence'] == {key:item for key,item in proof.items() if key != 'record'},
        'The reused mature mixed Muon evidence changed.')
    for item in value['record']['teacher_identity']['hf_shards']:
        require(binding(item['path']) == item, 'Original full-HF teacher shard changed: ' + item['path'])
    for name, digest in value['record']['teacher_identity']['hf_metadata_sha256'].items():
        require(mature.sha256(name) == digest, 'Original HF/tokenizer metadata changed: ' + name)
    parent, _meta = source(value['source_binding']['checkpoint_dir'], value['source_binding']['pipeline_report'])
    require(parent == value['source_binding'] and value['record']['source_checkpoint_id'] == parent['checkpoint_id'],
        'The actual C1 final identity changed after preparation.')
    return value


def verified_Rice_short(prepared_path):
    prepared_path = Path(prepared_path).resolve()
    report_path, audit_path = prepared_path.parent / 'short2/report.json', prepared_path.parent / 'short2/fresh_audit.json'
    report, audit = read(report_path), read(audit_path)
    require((report_path.parent / 'run.exit').read_text().strip() == '0'
        and report['schema'] == 'fresh_C1_Rice_Muon_execution_v1' and report['phase'] == 'short2'
        and report['validation_only'] is True and report['steps'] == 2 and report['group_indices'] == [0, 1]
        and report['status'] == 'Rice_Muon_training_export_complete_fresh_pending'
        and report['fresh_optimizer'] and report['original_schedule_steps'] == 500
        and audit['schema'] == 'fresh_C1_Rice_Muon_fresh_audit_v1' and audit['status'] == 'verified'
        and audit['phase'] == 'short2' and audit['validation_only'] is True and audit['steps'] == 2
        and audit['full_state_exact'] and audit['same_complete_input_BF16_logits_exact'] and audit['finite']
        and audit['Rice_frozen_exact'] and audit['training_report'] == binding(report_path)
        and report['prepared'] == audit['prepared'] == binding(prepared_path),
        'The new Rice two-update train/rollout/save/fresh/frozen path must pass before a fresh original-source32.')
    return dict(training=binding(report_path), fresh=binding(audit_path),
        source_checkpoint_id=audit['source_checkpoint_id'], checkpoint_id=audit['checkpoint_id'],
        scope='Engineering2 only; not counted in formal32 and never used as its model or optimizer source.')
