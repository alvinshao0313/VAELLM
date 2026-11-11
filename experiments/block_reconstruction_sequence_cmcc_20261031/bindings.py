"""Immutable wiring for one actual C0 -> BR36 -> Muon32 continuation."""
from pathlib import Path

from experiments.cmcc_runtime_20261030 import sequence as executor
from experiments.cmcc_runtime_20261030.paths import PYTHON, REPO, TOPIC, resolve_legacy_path
from experiments.fresh_muon_cmcc_20261030 import after_control as events

HERE = Path(__file__).resolve().parent
ROOT = TOPIC/'block_reconstruction_sequence_cmcc_20261031'
DIRECTORY = ROOT/'single_C0_v3'
CONTROL_PLAN = events.CONTROL_PLAN
MODEL_ROOT = events.ARTIFACTS/'block_reconstruction_sequence_cmcc_20261031'
BR, E = MODEL_ROOT/'BR36', MODEL_ROOT/'BR_E32'
BR_EVAL, E_EVAL = DIRECTORY/'evaluation_BR', DIRECTORY/'evaluation_BR_E'
SUPPORT = events.ARTIFACTS/'adaptive_class_cmcc_20261030/evaluation_support'
FORMAL = 'experiments.block_reconstruction_formal_cmcc_20261031.'
PACKAGE = 'experiments.block_reconstruction_sequence_cmcc_20261031.'
SCHEMA = 'actual_C0_v3_BR36_Muon32_single_sequence_v1'
BUDGET = dict(BR_layers=list(range(36)), BR_groups=list(range(32)), BR_updates_per_layer=32,
    BR_total_updates=1152, BR_lr=1e-6, E_groups=list(range(32)), E_updates=32,
    seed=137, original_schedule_steps=500, complete_nine_evaluations=2,
    training_gpus=[3], evaluation_gpus=[3, 4], additional_C0_or_direct_E_updates=0)
PROOF_ROOT = TOPIC/'block_reconstruction_formal_cmcc_20261031'
EVENT_PROOF = TOPIC/'fresh_muon_cmcc_20261030/after_control_cpu_v2/validation.json'
require, read, write, sha = executor.require, executor.read, executor.write, executor.sha


def c0_paths():
    control = read(CONTROL_PLAN)
    return Path(control['formal']['output'])/'final_model', Path(control['formal']['report'])


def requests():
    checkpoint, report = c0_paths()

    def row(name, module, arguments, gpu=''):
        return dict(name=name, argv=[str(PYTHON), '-B', '-u', '-m', module, *map(str, arguments)],
            environment=executor.environment(gpu), log=str(DIRECTORY/(name+'.log')),
            exit_file=str(DIRECTORY/(name+'.exit')))

    return [
        row('C0_source', PACKAGE+'audit', ['--kind', 'C0', '--checkpoint', checkpoint,
            '--report', report, '--output', DIRECTORY/'C0_source.json']),
        row('BR36', FORMAL+'run', ['--checkpoint', checkpoint, '--pipeline-report', report,
            '--output', BR, '--gpu', 3], '3'),
        row('BR_fresh', FORMAL+'fresh', ['--directory', BR, '--gpu', 3], '3'),
        row('BR_source', PACKAGE+'audit', ['--kind', 'BR', '--checkpoint', BR/'final_model',
            '--report', BR/'report.json', '--output', DIRECTORY/'BR_source.json']),
        row('E_prepare', FORMAL+'recovery_prepare', ['--checkpoint', BR/'final_model',
            '--br-report', BR/'report.json', '--output', E]),
        row('E32', FORMAL+'recovery_run', ['--prepared', E/'prepared.json', '--gpu', 3], '3'),
        row('E_fresh', FORMAL+'recovery_fresh', ['--prepared', E/'prepared.json', '--gpu', 3], '3'),
        row('BR_full9', FORMAL+'evaluate', ['--support', SUPPORT, '--checkpoint', BR/'final_model',
            '--pipeline-report', BR/'report.json', '--output', BR_EVAL, '--gpus', '3,4']),
        row('E_full9', FORMAL+'recovery_evaluate', ['--support', SUPPORT,
            '--checkpoint', E/'BR_E32/final_model', '--pipeline-report', E/'BR_E32/report.json',
            '--output', E_EVAL, '--gpus', '3,4']),
        row('comparison', PACKAGE+'summarize', []),
    ]


def bind_parent():
    control, run = read(CONTROL_PLAN), read(CONTROL_PLAN.parent/'run.json')
    require(control['schema'] == 'fresh_C0_fixed2_single_sequence_v1'
        and Path(control['output']).resolve(strict=True) == CONTROL_PLAN.parent
        and control['formal']['name'] == 'C0_fixed2' and control['formal']['gpus'] == ['5']
        and control['evaluation']['gpus'] == ['6', '7'], 'Use the actual fixed C0 v3 control plan.')
    stage = run['stages']['C0_fixed2']
    require((run['pid'], run['start_ticks']) == events.CONTROLLER
        and (stage['pid'], stage['start_ticks']) == events.WORKER
        and run['plan_sha256'] == sha(CONTROL_PLAN), 'The bound actual C0 controller/worker changed.')
    original = read(events.DIRECTORY/'plan.json')['parent']
    parent = dict(controller=original['controller'], worker=original['worker'],
        record_path=str(CONTROL_PLAN.parent/'run.json'), control_plan_sha256=sha(CONTROL_PLAN),
        stage_name='C0_fixed2', exit_file=control['formal']['exit_file'])
    require(parent == original and parent['worker']['argv'] == control['formal']['argv']
        and parent['worker']['parent_pid'] == parent['controller']['pid'], 'Keep the real previously bound process commands.')
    events.formal_record(parent)
    # Attaching the identities is read-only and does not wait for either process.
    for identity in (parent['controller'], parent['worker']):
        descriptor = events.open_identity(identity)
        if descriptor is not None:
            import os
            os.close(descriptor)
    return parent


def proofs_and_pins():
    from experiments.block_reconstruction_formal_cmcc_20261031 import recovery_bindings as recovery
    from experiments.block_reconstruction_formal_cmcc_20261031 import bindings as formal
    formal.configure_readers()
    event = read(EVENT_PROOF)
    require(event['status'] == 'passed' and event['tests_run'] == 9 and not event['failures']
        and not event['errors'] and event['real_CPU_children_and_kernel_pidfd_inotify'],
        'Reuse the completed real nine-test pidfd/inotify proof.')
    executor.check_pins(event['implementation_sha256'])
    formal.verified_short()
    recovery.verified_implementation()
    proofs = [EVENT_PROOF, PROOF_ROOT/'cpu_v1/validation.json', PROOF_ROOT/'recovery_cpu/validation.json',
        PROOF_ROOT/'readers_cpu/validation.json']
    pins = dict(read(events.DIRECTORY/'plan.json')['source_sha256'])
    for path, item in recovery.code_bindings().items():
        require(path not in pins or pins[path] == item['sha256'], 'Conflicting existing source pin: '+path)
        pins[path] = item['sha256']
    for path in [*HERE.glob('*.py'), Path(events.__file__), *proofs, CONTROL_PLAN,
                 events.DIRECTORY/'plan.json', SUPPORT/'prepared.json']:
        key, digest = str(path.resolve()), sha(path)
        require(key not in pins or pins[key] == digest, 'Conflicting continuation source pin: '+key)
        pins[key] = digest
    support = read(SUPPORT/'prepared.json')
    for path, digest in support['source_sha256'].items():
        require(path not in pins or pins[path] == digest, 'Conflicting evaluation source pin: '+path)
        pins[path] = digest
    inputs = {path:item['sha256'] for path,item in read(proofs[1])['data']['bindings'].items()}
    target_manifest = recovery.inputs.HOT/'teacher_targets.json'
    inputs[str(target_manifest)] = sha(target_manifest)
    for component in read(target_manifest)['components']:
        for key in ('cache_binding', 'payload_binding'):
            item = component[key]
            path = resolve_legacy_path(item['path']).resolve(strict=True)
            require(path.stat().st_size == item['bytes'], 'Actual target input size changed: '+str(path))
            inputs[str(path)] = item['sha256']
    executor.check_pins(pins)
    executor.check_pins(inputs)
    return pins, inputs, [dict(path=str(path), sha256=sha(path)) for path in proofs]
