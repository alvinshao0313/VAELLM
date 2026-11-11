"""CPU math and actual-tokenizer train-row tests; never an LLM/GPU pilot."""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as d
from . import proposal_readout as p
from .contracts import digest, require


def loss_math():
    from train_utils.distill_loss_core import compute_model_level_loss, compute_kl_top_mass_token_loss

    generator = torch.Generator().manual_seed(1307)
    rows = []
    for vocab, dtype in ((137, torch.float32), (97, torch.float32), (151936, torch.bfloat16)):
        teacher = torch.randn(2, 7, vocab, generator=generator).to(dtype)
        student = torch.randn(2, 7, vocab, generator=generator).to(dtype)
        ids = torch.randint(vocab, (2, 7), generator=generator)
        labels = ids.clone()
        labels[:, :3] = -100
        attention = torch.ones_like(ids)
        attention[1, -2:] = 0
        labels[1, -2:] = -100
        inputs = dict(input_ids=ids, labels=labels, attention_mask=attention)
        target = p.pack_teacher(teacher)
        cached = p.cached_token_kl(student, target)
        original = compute_kl_top_mass_token_loss(student_logits=student[:, :-1],
            teacher_logits=teacher[:, :-1], temperature=1., top_k=100)
        torch.testing.assert_close(cached, original, rtol=2e-5, atol=3e-6)
        numerator, denominator = p.weighted_rows(cached, inputs)
        expected = compute_model_level_loss(loss_type='kl_top_mass', student_logits=student,
            teacher_logits=teacher, input_ids=ids, labels=labels, attention_mask=attention,
            temperature=1., top_k=100, prompt_loss_weight=.3)
        actual = numerator.sum() / denominator.sum()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        broken = copy.deepcopy(target)
        broken['top_logp'][0, 0, 0] += .125
        try:
            p.cached_token_kl(student, broken)
        except ValueError:
            pass
        else:
            raise AssertionError('A changed teacher target was accepted.')
        rows.append(dict(vocab=vocab, logits_dtype=str(dtype), maximum_token_error=float((cached - original).abs().max()),
            pooled_error=float((actual - expected).abs()), effective_token_weight=float(denominator.sum())))
    pooled = [dict(record_id=str(index), alias='edgerazor_ii_7m' if index < 384 else 'vaellm_eval_task',
        numerator=1. if index < 384 else 20., denominator=1. if index < 384 else 10.) for index in range(512)]
    aggregate = p.aggregate_rows(pooled)
    require(abs(aggregate['full']['KL'] - 2944 / 1664) < 1e-14,
        'Do not change sampled-record weights into an additional per-source objective.')
    require(aggregate['per_source']['edgerazor_am']['observed'] is None, 'Missing domains are diagnostic, not a reweighting rule.')
    return dict(cases=rows, unequal_source_counts_and_token_lengths=aggregate['full'],
        missing_domains_diagnostic_only=True, cache_tamper_rejected=True)


def eval_lifecycle():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.5), torch.nn.LayerNorm(4))
    model.train()
    model[1].eval()
    modes = [module.training for module in model.modules()]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with p.preserved_eval(model):
        require(not any(module.training for module in model.modules()), 'Eval context did not cover all actual modules.')
        random.random()
        np.random.random()
        torch.rand(3)
        model(torch.ones(2, 3))
    require(modes == [module.training for module in model.modules()], 'Module-specific mode overrides were lost.')
    require(py_rng == random.getstate() and torch.equal(torch_rng, torch.get_rng_state()), 'Readout consumed a training random stream.')
    after_numpy = np.random.get_state()
    require(np_rng[0] == after_numpy[0] and np.array_equal(np_rng[1], after_numpy[1])
        and np_rng[2:] == after_numpy[2:], 'Readout consumed NumPy state.')
    require(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()), 'Eval context changed module state.')
    return dict(scope='Actual PyTorch module context only, not native model resource/gradient proof',
        modes_exact=True, rng_exact=True, state_exact=True)


def real_source(tokenizer_path):
    from e2e_common.data import build_tokenizer, DATASET_MIX_SOURCE_PRESETS, _load_preset_raw_datasets
    from e2e_common.lazy_datasets import build_calibration_input_ids_lazy
    from train_utils.config.configs import DistillDataConfig
    from train_utils.distill_data import build_distill_dataset, build_distill_data_collator

    tokenizer = build_tokenizer(str(tokenizer_path), None)
    alias = 'vaellm_eval_task'
    source_path = Path('data/edgerazor_qwen3/task_vaellm_eval_instruct.jsonl').resolve()
    expected_sha = '894a04aea907efebd8b0f52f040c5635750bb30dfb64d9c34e30a97421a155a1'
    require(d.file_sha256(source_path) == expected_sha, 'The actual migrated task train source differs.')
    raw, _unused_eval = _load_preset_raw_datasets(DATASET_MIX_SOURCE_PRESETS[alias])
    source = d.bind_source(raw, DATASET_MIX_SOURCE_PRESETS[alias],
        revision='converted-artifact-sha256:' + expected_sha,
        license_record=dict(scope='CPU mechanism test; full upstream license registry still requires completion'),
        provenance_record=dict(split='train', generator='tools/prepare_vaellm_task_mix.py',
            evidence='docs/experience/lessons/kd_and_data.md#真实artifactsplit与train-only替代关系'),
        artifact_digest=expected_sha)
    calibration, record = d.calibration_blocks(source, tokenizer)
    original = build_calibration_input_ids_lazy('vaellm_eval_task=1', tokenizer=tokenizer,
        nsamples=512, seqlen=256, seed=31)
    require(all(torch.equal(left, right) for left, right in zip(calibration, original)),
        'Raw-row provenance changed the actual canonical packed calibration tokens.')
    model_identity = dict(scope='CPU data test, not formal run identity', tokenizer=d.tokenizer_manifest(tokenizer))
    traced = d.trace_updates([source], tokenizer, model_identity=model_identity,
        category_id='q_proj', search_seed=31, steps=500, workers=16)
    cfg = DistillDataConfig(dataset_mix='vaellm_eval_task=1', dataset_task='sft', model_max_length=1024,
        data_seed=traced['data_seed'], group_by_length=True)
    bundle = build_distill_dataset(cfg, tokenizer)
    require(bundle.is_iterable and not bundle.group_by_length, 'Keep actual canonical iterable sampling.')
    loader = DataLoader(bundle.train_dataset, batch_size=4, num_workers=16,
        collate_fn=build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True),
        generator=torch.Generator().manual_seed(traced['data_seed']), pin_memory=False)
    iterator = iter(loader)
    observer = d.UpdateObserver(traced)
    try:
        for index in range(32):
            inputs = next(iterator)
            require(d.batch_digest(inputs) == traced['batches'][index]['input_sha256'],
                'Tagged data changed actual16-worker consumption/padding order.')
            observer(inputs, index)
    finally:
        iterator._shutdown_workers()
    excluded = {ref['record_id'] for batch in traced['batches'] for ref in batch['records']}
    excluded.update(ref['record_id'] for refs in record['records_by_block'] for ref in refs)
    samples, manifest = d.select_proposal([source], tokenizer, model_identity=model_identity,
        search_seed=31, excluded_ids=excluded)
    require(len(samples) == len({sample[d.ROW_KEY]['record_id'] for sample in samples}) == 512
        and not ({sample[d.ROW_KEY]['record_id'] for sample in samples} & excluded), 'Proposal ID exclusion failed.')
    batches = p.proposal_batches(samples, tokenizer)
    require(len(batches) == 128 and all(len(batch['records']) == 4 for batch in batches), 'Keep actualB4 proposal batching.')
    return dict(actual_source=source.identity, source_sha256=expected_sha, rows=len(raw),
        tokenizer=d.tokenizer_manifest(tokenizer), packed_calibration512_exact=True,
        actual_workers=16, traced_batch_count=500, original_batch_parity_count=32,
        reserve_record_presentations=2000, observer=observer.report(),
        proposal=manifest, source_ID_exclusion_exact=True,
        verification_scope='One actual registered train source and real tokenizer; full five-source root plan/LLM forwards not run')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), 'Never overwrite an earlier CPU validation.')
    require(not torch.cuda.is_initialized(), 'This validation must remain CPU-only.')
    torch.set_num_threads(4)
    began = time.monotonic()
    result = dict(loss_math=loss_math(), eval_lifecycle=eval_lifecycle(), actual_source=real_source(args.tokenizer))
    require(not torch.cuda.is_initialized(), 'CPU mechanism checks initialized CUDA.')
    result.update(status='completed_cpu_data_mechanisms', exit_code=0, cuda_initialized=False,
        elapsed_seconds=time.monotonic() - began, formal_model_or_runtime_verified=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'exit_code', 'elapsed_seconds', 'cuda_initialized')}), flush=True)


if __name__ == '__main__':
    main()
