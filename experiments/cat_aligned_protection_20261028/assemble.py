"""04 C1: whole Q/K/V/O from actual03b64, all other projections from actual03a32."""
import argparse
import gc
from pathlib import Path
import bindings as b
from deployment import save_and_verify
from experiments.w25_eval.assemble_categories import validate_sources
from experiments.qwen3_w25.assemble_categories import _expected_state, _replace_modules, _assert_state_equal
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
from train_utils.v6_model_loader import load_v6_model_checkpoint

CATEGORIES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']


def arm_source(arm, prepared):
    root = b.OUTPUT / arm
    audit = b.read(root / 'native_audit.json')
    report_path = root / 'formal/report.json'
    report = b.read(report_path)
    directory = root / 'native_final/final_model'
    meta = b.read(directory / 'checkpoint_meta.json')
    b.require(audit['schema'] == 'CG_aligned_protection_native_audit_v1' and audit['status'] == 'verified'
        and audit['profile'] == report['profile'] == b.PROFILE and audit['arm'] == report['arm'] == arm
        and audit['phase'] == report['phase'] == 'formal' and audit['source_checkpoint_id'] == prepared['source_checkpoint_id']
        and audit['actual_steps'] == audit['planned_steps'] == report['steps'] == b.ARMS[arm]['steps']
        and audit['checkpoint_dir'] == str(directory) and audit['checkpoint_id'] == meta['checkpoint_id']
        and audit['endpoint_report'] == b.binding(report_path) and (report_path.parent / 'run.exit').read_text().strip() == '0'
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and audit['fresh_strict_reload_state_exact'] and audit['code_bank_count'] == 504 and audit['bit_budget']['budget_passed']
        and report['prepared_sha256'] == b.sha256(b.OUTPUT / 'prepared.json')
        and report['status'] == 'aligned_protection_formal_complete', 'C1 requires both same-profile actual complete03 arms from identical new02 parent.')
    b.check_hashes(audit['sources']['input_evidence_sha256'])
    b.check_stat(audit['checkpoint_weight_stat'])
    b.require(audit['checkpoint_meta_sha256'] == b.sha256(directory / 'checkpoint_meta.json'), 'Changed03 native source metadata.')
    return directory, meta, audit


@b.torch.no_grad()
def assemble():
    prepared = b.read_prepared()
    base_dir, base_meta, base_audit = arm_source('scale32', prepared)
    donor_dir, donor_meta, donor_audit = arm_source('q64', prepared)
    output = b.ASSEMBLY / 'final_model'
    b.require(not output.exists() and not output.is_symlink(), 'Never replace actual04 C1.')
    bm, dm, names, architecture, tokenizer, runtime, differences = validate_sources(base_dir, donor_dir, CATEGORIES)
    b.require(bm == base_meta and dm == donor_meta and len(names) == 144
        and {_['name']: _inference_spec(_) for _ in bm['converted_modules']}
        == {_['name']: _inference_spec(_) for _ in dm['converted_modules']}, 'Same-parent03 geometry must match exactly.')
    base_state, donor_state = b.load_state(base_dir, bm), b.load_state(donor_dir, dm)
    b.require(_assert_state_equal(b.aligned02.outside_projections(base_state, architecture['projection_shapes']),
        b.aligned02.outside_projections(donor_state, architecture['projection_shapes']), 'Exact147 nonprojection tensors') == 147,
        'C1 cannot cross backbone/profile.')
    expected = _expected_state(base_state, donor_state, names, architecture['projection_shapes'])
    specs = {spec['name']: spec for spec in dm['converted_modules']}
    assembled_meta = {**bm, 'converted_modules': [specs[spec['name']] if spec['name'] in names else spec for spec in bm['converted_modules']]}
    base, loaded, result = load_v6_model_checkpoint(str(base_dir), map_location='cpu', strict=True, expected_kind='final_model')
    b.require(loaded == bm and not result.missing_keys and not result.unexpected_keys, 'Strict03a loading failed.')
    donor, loaded, result = load_v6_model_checkpoint(str(donor_dir), map_location='cpu', strict=True, expected_kind='final_model')
    b.require(loaded == dm and not result.missing_keys and not result.unexpected_keys, 'Strict03b loading failed.')
    _replace_modules(base, donor, names)
    _assert_state_equal(base.state_dict(), expected, 'C1 whole projection D/banks/q/scales/indices/rotation')
    del donor
    gc.collect()
    provenance = dict(parent_checkpoint_ids=[bm['checkpoint_id'], dm['checkpoint_id']], profile=b.PROFILE,
        aligned02_parent_checkpoint_id=prepared['source_checkpoint_id'], categories=CATEGORIES, replaced_targets=names,
        base_checkpoint_dir=str(base_dir), base_checkpoint_id=bm['checkpoint_id'], base_arm='scale32', base_actual_steps=32,
        donor_checkpoint_dir=str(donor_dir), donor_checkpoint_id=dm['checkpoint_id'], donor_arm='q64', donor_actual_steps=64,
        base_audit=b.binding(b.OUTPUT / 'scale32/native_audit.json'), donor_audit=b.binding(b.OUTPUT / 'q64/native_audit.json'),
        input_evidence_sha256={str(path): b.sha256(path) for path in (b.OUTPUT / 'prepared.json',
            b.OUTPUT / 'scale32/native_audit.json', b.OUTPUT / 'q64/native_audit.json',
            b.OUTPUT / 'scale32/formal/report.json', b.OUTPUT / 'q64/formal/report.json')},
        source_config_runtime_fields=runtime, selected_spec_differences=differences, nonprojection_tensors_compared=147,
        purpose='C1 inference-only complete144 attention projection replacement; D/banks/q/scales/indices/rotation move together; noC2/search.')
    report = save_and_verify(base, output, assembled_meta, expected, provenance, tokenizer)
    report.update(schema='CG_aligned_C1_native_assembly_v1', nonprojection_tensors_compared=147, replaced_projection_count=144)
    b.core.write(b.ASSEMBLY / 'assembly_report.json', report)
    return report


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    b.torch.set_num_threads(4)
    print(assemble(), flush=True)
