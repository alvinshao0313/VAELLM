"""Small-tensor formula checks; never a substitute for native GPU validation."""
from __future__ import annotations

import time

import torch

from experiments.llama_column_feedback_probe.algorithm import ARMS, choices_to_codes, scalar_ldl, search_columns
from experiments.qwen3_w25.block_ldlq import block_ldl, two_stage_single_bit_candidates
from experiments.rotation_factor_probe.capture import require
from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool


def run_checks():
    require(not torch.cuda.is_initialized(), 'Formula checks must remain on CPU.')
    generator = torch.Generator().manual_seed(1060616)
    x = torch.randn((80, 32), generator=generator, dtype=torch.float64)
    h = x.T @ x / len(x) + torch.eye(32, dtype=torch.float64) * .01
    lower, diagonal = scalar_ldl(h)
    old = block_ldl(h, block_size=1)
    require(torch.allclose(lower, old.lower, atol=1e-13, rtol=1e-13)
            and torch.allclose(diagonal, old.diagonal.flatten(), atol=1e-13, rtol=1e-13),
            'Vectorized scalar LDL differs from the existing block_size=1 factorization.')
    factor_error = float((lower @ torch.diag(diagonal) @ lower.T - h).abs().max())
    require(factor_error < 1e-12, 'Scalar LDL failed to reproduce H.')
    codes = torch.randint(0, 2, (2, 32, 2, 24), generator=generator).bool()
    weights = torch.randn((2, 24, 16), generator=generator)
    # Synthetic finite quantizer for algebra only; native fused BF16 decoder remains untested here.
    def decode(stage, bits):
        return ((bits.float() * 2 - 1) @ weights[stage]).to(torch.bfloat16)
    target = torch.randn((32, 32), generator=generator, dtype=torch.float64) * 5
    choices, values, initial, statistics = search_columns(target, lower, diagonal, codes, decode,
                                                         deadline=time.time() + 60)
    all_candidates = two_stage_single_bit_candidates(codes.reshape(2, 64, 24), decode).reshape(32, 2, 49, 16)
    naive = {arm: torch.empty_like(target, dtype=torch.bfloat16) for arm in ARMS}
    naive_choices = {arm: torch.zeros((32, 2), dtype=torch.uint8) for arm in ARMS}
    for arm in ARMS:
        for j in range(31, -1, -1):
            first = j + 1 if arm == 'per_column' else (j // 16 + 1) * 16
            adjusted = target[:, j].clone()
            if arm != 'off' and first < 32:
                adjusted += (target[:, first:] - naive[arm][:, first:].double()) @ lower[first:, j]
            scores = (all_candidates[j].double() - adjusted.reshape(2, 1, 16)).square().sum(-1) * diagonal[j]
            index = scores.argmin(-1)
            naive_choices[arm][j] = index.to(torch.uint8)
            naive[arm][:, j] = all_candidates[j, torch.arange(2), index].reshape(32)
        require(torch.equal(choices[arm], naive_choices[arm]) and torch.equal(values[arm], naive[arm]),
                'Lazy implementation differs from the explicit original-residual formula: ' + arm)
        replay = choices_to_codes(codes, choices[arm])
        for stage in range(2):
            logical = replay[stage].reshape(64, 1, 24)
            packed = pack_bool_tensor_to_uint8(logical, logical_shape=logical.shape)
            require(torch.equal(unpack_uint8_tensor_to_bool(packed, logical_shape=logical.shape), logical),
                    'Native packed-byte replay changed a selected logical bit.')
        decoded = (decode(0, replay[0].reshape(-1, 24)) + decode(1, replay[1].reshape(-1, 24))).reshape(32, 32).T
        require(torch.equal(decoded, values[arm]), 'Flat transposed-bank indexing or choice replay is incorrect.')
        diff = target - values[arm].double()
        direct = float(((diff @ h) * diff).sum())
        require(abs(direct - statistics[arm]['full_damped_protection_compensated_proxy']) <= 1e-10 * max(1, abs(direct)),
                'LDL proxy differs from the direct full-H quadratic.')
    difference = int((choices['per_column'] != choices['delayed16']).sum())
    require(difference > 0, 'The synthetic intervention failed to distinguish intentional delay from immediate feedback.')
    require(torch.equal(initial, all_candidates[:, :, 0].reshape(32, 32).T), 'Original candidate layout differs.')
    from experiments.gate_tail_objective_probe.core import probability_metrics, teacher_statistics
    from train_utils.distill_loss_core import compute_kl_top_mass_token_loss
    for dtype in (torch.float32, torch.bfloat16):
        teacher = torch.randn((7, 131), generator=generator).to(dtype)
        student = torch.randn((7, 131), generator=generator).to(dtype)
        labels = torch.arange(7)
        actual_kl, actual_nll = probability_metrics(student, teacher_statistics(teacher, labels))
        native_kl = compute_kl_top_mass_token_loss(student_logits=student, teacher_logits=teacher,
                                                  temperature=1.0, top_k=100)
        require(torch.equal(actual_kl, native_kl)
                and torch.equal(actual_nll, torch.nn.functional.cross_entropy(student.float(), labels, reduction='none')),
                'Cached teacher statistics differ from native token KL/NLL.')
    return {'status': 'cpu_formula_verified', 'cuda_initialized': torch.cuda.is_initialized(),
            'scalar_ldl_max_abs_error': factor_error, 'scalar_ldl_matches_existing_block1': True,
            'all_three_lazy_formulas_match_explicit_reverse_sums': True,
            '49_candidate_flat_transpose_choice_replay_exact': True,
            'native_uint8_pack_replay_exact': True, 'cached_kl_nll_matches_native_fp32_bf16': True,
            'full_h_proxy_matches_ldl_decomposition': True, 'delayed16_vs_column_choice_differences': difference,
            'native_gpu_decoder_capture_save_and_quality_tested': False}
