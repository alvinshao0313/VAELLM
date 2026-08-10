import unittest
from unittest import mock

import torch
from torch import nn

from compressed_e2e_fintuning.offload import OffloadedCheckpointLayer
from compressed_e2e_fintuning.teacher_targets import (
    StudentHiddenCollector,
    TeacherHiddenTargetCollector,
    TeacherTargetBatch,
    copy_detached_tensor_to_cpu,
    copy_teacher_logit_chunk_to_device,
    extract_primary_hidden,
    iter_token_chunk_ranges,
    resolve_transformer_layers,
)
from train_utils.lora_training import (
    _masked_mean_cosine_similarity,
    _select_adaptive_hidden_layer_indices,
    compute_distill_hidden_alignment_loss,
    compute_masked_hidden_transition_cosine,
    compute_selected_distill_hidden_alignment_loss,
)


class _ModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module


class _BaseModelWrapper(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self._base = base

    def get_base_model(self) -> nn.Module:
        return self._base


class _FakeOffloadManager:
    def __init__(self) -> None:
        self.registered = []

    def register(self, wrapper) -> None:
        self.registered.append(wrapper)


class _TinyModel(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Linear(4, 4, bias=False) for _ in range(num_layers)]
        )


class TeacherTargetOffloadUtilityTest(unittest.TestCase):
    def test_copy_detached_tensor_to_cpu_preserves_values_and_removes_grad(self):
        source = torch.randn(2, 3, requires_grad=True)
        if torch.cuda.is_available():
            source = source.cuda()

        copied = copy_detached_tensor_to_cpu(source, pin_memory=False)

        self.assertEqual(copied.device.type, "cpu")
        self.assertFalse(copied.requires_grad)
        self.assertEqual(copied.dtype, source.dtype)
        self.assertTrue(torch.equal(copied, source.detach().cpu()))

    def test_copy_detached_tensor_to_cpu_from_cpu_input(self):
        source = torch.randn(2, 3)
        copied = copy_detached_tensor_to_cpu(source, pin_memory=False)
        self.assertEqual(copied.device.type, "cpu")
        self.assertFalse(copied.requires_grad)
        self.assertTrue(torch.equal(copied, source))

    def test_iter_token_chunk_ranges_L11_chunk4(self):
        ranges = list(iter_token_chunk_ranges(11, 4))
        self.assertEqual(ranges, [(0, 4), (4, 8), (8, 11)])

    def test_iter_token_chunk_ranges_invalid_raises(self):
        with self.assertRaises(ValueError):
            list(iter_token_chunk_ranges(0, 4))
        with self.assertRaises(ValueError):
            list(iter_token_chunk_ranges(11, 0))

    def test_copy_teacher_logit_chunk_to_device_respects_range(self):
        logits_cpu = torch.randn(2, 11, 32)
        target_device = torch.device("cpu")
        if torch.cuda.is_available():
            target_device = torch.device("cuda")

        chunk = copy_teacher_logit_chunk_to_device(
            logits_cpu,
            start=4,
            end=8,
            target_device=target_device,
        )

        self.assertEqual(tuple(chunk.shape), (2, 4, 32))
        self.assertTrue(torch.equal(chunk.cpu(), logits_cpu[:, 4:8, :]))

    def test_copy_teacher_logit_chunk_to_device_invalid_range_raises(self):
        logits_cpu = torch.randn(2, 11, 32)
        with self.assertRaises(ValueError):
            copy_teacher_logit_chunk_to_device(
                logits_cpu,
                start=8,
                end=4,
                target_device=torch.device("cpu"),
            )

    def test_teacher_target_batch_clear(self):
        batch = TeacherTargetBatch(
            logits_cpu=torch.randn(2, 4, 8),
            eakld_gamma_cpu=torch.randn(2, 4),
            teacher_entropy_mean_cpu=torch.tensor(1.5),
            teacher_valid_token_count_cpu=torch.tensor(8.0),
            hidden_cpu_by_layer={0: torch.randn(2, 4, 16)},
            hidden_layer_indices=(0,),
            num_hidden_layers=1,
        )

        batch.clear()

        self.assertIsNone(batch.logits_cpu)
        self.assertIsNone(batch.eakld_gamma_cpu)
        self.assertIsNone(batch.teacher_entropy_mean_cpu)
        self.assertIsNone(batch.teacher_valid_token_count_cpu)
        self.assertEqual(batch.hidden_cpu_by_layer, {})
        self.assertEqual(batch.hidden_layer_indices, ())
        self.assertEqual(batch.num_hidden_layers, 0)

    @mock.patch("compressed_e2e_fintuning.teacher_targets.get_layers")
    def test_resolve_transformer_layers_unwraps_wrappers(self, mock_get_layers):
        tiny_model = _TinyModel(num_layers=2)
        mock_get_layers.side_effect = lambda model: model.model.layers

        module_wrapped = _ModuleWrapper(tiny_model)
        resolved = resolve_transformer_layers(module_wrapped)
        self.assertEqual(len(resolved), 2)
        self.assertIs(resolved[0], tiny_model.model.layers[0])
        mock_get_layers.assert_called_once_with(tiny_model)

        mock_get_layers.reset_mock()
        base_wrapped = _BaseModelWrapper(tiny_model)
        resolved = resolve_transformer_layers(base_wrapped)
        self.assertEqual(len(resolved), 2)
        mock_get_layers.assert_called_once_with(tiny_model)

        mock_get_layers.reset_mock()
        double_wrapped = _ModuleWrapper(_BaseModelWrapper(tiny_model))
        resolved = resolve_transformer_layers(double_wrapped)
        self.assertEqual(len(resolved), 2)
        mock_get_layers.assert_called_once_with(tiny_model)

    @mock.patch("compressed_e2e_fintuning.teacher_targets.get_layers")
    def test_resolve_transformer_layers_offloaded_layer(self, mock_get_layers):
        tiny_model = _TinyModel(num_layers=1)
        inner_layer = nn.Linear(4, 4, bias=False)
        manager = _FakeOffloadManager()
        wrapped_layer = OffloadedCheckpointLayer(
            layer=inner_layer,
            layer_idx=0,
            manager=manager,
        )
        tiny_model.model.layers = nn.ModuleList([wrapped_layer])
        mock_get_layers.side_effect = lambda model: model.model.layers

        resolved = resolve_transformer_layers(tiny_model)
        self.assertEqual(resolved, (inner_layer,))

    def test_extract_primary_hidden_tensor(self):
        tensor = torch.randn(2, 3)
        self.assertIs(extract_primary_hidden(tensor, context="test"), tensor)

    def test_extract_primary_hidden_tuple(self):
        tensor = torch.randn(2, 3)
        self.assertIs(extract_primary_hidden((tensor, torch.randn(2, 3)), context="test"), tensor)

    def test_extract_primary_hidden_invalid_raises(self):
        with self.assertRaises(TypeError):
            extract_primary_hidden(42, context="test")


class _AddDeltaBlock(nn.Module):
    def __init__(self, delta: float, hidden_size: int = 8) -> None:
        super().__init__()
        self.register_buffer(
            "delta",
            torch.full((1, 1, hidden_size), float(delta)),
        )

    def forward(self, hidden_states):
        return hidden_states + self.delta


class _DeltaStackModel(nn.Module):
    def __init__(self, deltas) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_AddDeltaBlock(delta) for delta in deltas]
        )

    def forward(self, hidden_states):
        current = hidden_states
        for layer in self.model.layers:
            current = layer(current)
        return current


class _TrackingLinear(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_output = None

    def forward(self, hidden_states):
        out = self.linear(hidden_states)
        self.last_output = out
        return out


class _TrackingStackModel(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_size: int = 4) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_TrackingLinear(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, hidden_states):
        current = hidden_states
        for layer in self.model.layers:
            current = layer(current)
        return current


class HiddenTransitionCosineTest(unittest.TestCase):
    def test_chunked_matches_full_tensor_helper(self):
        torch.manual_seed(0)
        input_hidden = torch.randn(2, 17, 8)
        output_hidden = input_hidden + 0.25 * torch.randn_like(input_hidden)
        attention_mask = torch.ones(2, 17)
        attention_mask[0, -3:] = 0
        reference = _masked_mean_cosine_similarity(
            output_hidden, input_hidden, attention_mask
        )
        for chunk_size in (1, 2, 16):
            chunked = compute_masked_hidden_transition_cosine(
                input_hidden=input_hidden,
                output_hidden=output_hidden,
                attention_mask=attention_mask,
                sequence_chunk_size=chunk_size,
            )
            self.assertTrue(torch.allclose(chunked, reference, atol=1e-6, rtol=1e-5))

    def test_shape_and_chunk_validation(self):
        a = torch.randn(2, 4, 8)
        b = torch.randn(2, 4, 8)
        with self.assertRaises(ValueError):
            compute_masked_hidden_transition_cosine(
                input_hidden=a,
                output_hidden=torch.randn(2, 5, 8),
                attention_mask=None,
            )
        with self.assertRaises(ValueError):
            compute_masked_hidden_transition_cosine(
                input_hidden=torch.randn(2, 4),
                output_hidden=b,
                attention_mask=None,
            )
        with self.assertRaises(ValueError):
            compute_masked_hidden_transition_cosine(
                input_hidden=a,
                output_hidden=b,
                attention_mask=None,
                sequence_chunk_size=0,
            )


class TeacherHiddenTargetCollectorTest(unittest.TestCase):
    @mock.patch("compressed_e2e_fintuning.teacher_targets.get_layers")
    def test_adaptive_top2_matches_reference_selection(self, mock_get_layers):
        model = _DeltaStackModel(deltas=(0.1, 0.8, 0.05, 1.5))
        mock_get_layers.side_effect = lambda m: m.model.layers
        hidden = torch.randn(2, 5, 8)
        attention_mask = torch.ones(2, 5)

        reference_outputs = {}
        reference_scores = []
        current = hidden
        for layer_idx, layer in enumerate(model.model.layers):
            layer_input = current
            layer_output = layer(layer_input)
            reference_outputs[layer_idx] = layer_output.detach().clone()
            score = float(
                _masked_mean_cosine_similarity(
                    layer_output, layer_input, attention_mask
                ).item()
            )
            reference_scores.append((score, layer_idx))
            current = layer_output

        expected = tuple(
            layer_idx
            for _score, layer_idx in sorted(reference_scores)[:2]
        )

        collector = TeacherHiddenTargetCollector(
            model=model,
            attention_mask=attention_mask,
            layer_weighting="adaptive_top_2",
            pin_memory=False,
            score_chunk_tokens=2,
        )
        with collector:
            _ = model(hidden)
        selected_ids, hidden_by_layer, num_layers = collector.finalize()

        self.assertEqual(selected_ids, expected)
        self.assertEqual(num_layers, 4)
        self.assertEqual(set(hidden_by_layer.keys()), set(expected))
        self.assertLessEqual(collector._max_retained_hidden_count, 2)
        for layer_id in expected:
            cached = hidden_by_layer[layer_id]
            self.assertEqual(cached.device.type, "cpu")
            self.assertFalse(cached.requires_grad)
            self.assertTrue(torch.equal(cached, reference_outputs[layer_id].cpu()))


class StudentHiddenCollectorTest(unittest.TestCase):
    @mock.patch("compressed_e2e_fintuning.teacher_targets.get_layers")
    def test_selected_layers_preserve_identity_and_grad(self, mock_get_layers):
        model = _TrackingStackModel(num_layers=4, hidden_size=4)
        mock_get_layers.side_effect = lambda m: m.model.layers
        x = torch.randn(2, 3, 4, requires_grad=True)

        collector = StudentHiddenCollector(model=model, layer_indices=(1, 3))
        with collector:
            _ = model(x)
        captured = collector.collected()

        self.assertEqual(set(captured.keys()), {1, 3})
        for layer_id in (1, 3):
            tensor = captured[layer_id]
            self.assertTrue(tensor.requires_grad)
            self.assertEqual(
                tensor.data_ptr(),
                model.model.layers[layer_id].last_output.data_ptr(),
            )

        loss = captured[1].pow(2).mean() + captured[3].pow(2).mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for layer in model.model.layers:
            weight_grad = layer.linear.weight.grad
            if weight_grad is not None:
                self.assertTrue(torch.isfinite(weight_grad).all())
        self.assertIsNotNone(model.model.layers[1].linear.weight.grad)
        self.assertIsNotNone(model.model.layers[3].linear.weight.grad)

    @mock.patch("compressed_e2e_fintuning.teacher_targets.get_layers")
    def test_missing_or_repeated_layer_raises(self, mock_get_layers):
        model = _TrackingStackModel(num_layers=4, hidden_size=4)
        mock_get_layers.side_effect = lambda m: m.model.layers
        x = torch.randn(2, 3, 4)

        collector = StudentHiddenCollector(model=model, layer_indices=(1, 3))
        with self.assertRaises(RuntimeError):
            with collector:
                current = x
                for layer_id in (0, 1, 2):
                    current = model.model.layers[layer_id](current)
            collector.collected()

        collector = StudentHiddenCollector(model=model, layer_indices=(1,))
        with self.assertRaises(RuntimeError):
            with collector:
                _ = model.model.layers[1](x)
                _ = model.model.layers[1](x)


class SelectedHiddenAlignmentLossTest(unittest.TestCase):
    def _make_distinct_score_sequences(self):
        torch.manual_seed(123)
        batch, seq, hidden = 2, 6, 8
        teacher = [torch.randn(batch, seq, hidden) for _ in range(5)]
        # Amplify block deltas so adaptive cosine scores stay distinct.
        scales = (0.05, 0.9, 0.2, 1.7)
        for idx, scale in enumerate(scales):
            teacher[idx + 1] = teacher[idx] + scale * torch.randn(batch, seq, hidden)
        student = [t + 0.1 * torch.randn_like(t) for t in teacher]
        student = [
            tensor.clone().detach().requires_grad_(True) if i > 0 else tensor
            for i, tensor in enumerate(student)
        ]
        attention_mask = torch.ones(batch, seq)
        attention_mask[0, -2:] = 0
        return teacher, student, attention_mask

    def test_selected_api_matches_reference_for_modes(self):
        teacher, student, attention_mask = self._make_distinct_score_sequences()
        num_layers = 4

        for layer_weighting in ("adaptive_top_2", "uniform", "linear_depth"):
            student_blocks = [tensor.clone().detach().requires_grad_(True) for tensor in student[1:]]
            student_states = [student[0]] + student_blocks
            reference = compute_distill_hidden_alignment_loss(
                teacher_hidden_states=teacher,
                student_hidden_states=student_states,
                attention_mask=attention_mask,
                layer_weighting=layer_weighting,
            )
            reference.backward()
            reference_grads = [block.grad.detach().clone() for block in student_blocks]

            if layer_weighting.startswith("adaptive"):
                selected = tuple(
                    _select_adaptive_hidden_layer_indices(
                        teacher[1:],
                        attention_mask,
                        topk=2,
                        reference_hidden=teacher[0],
                    )
                )
            else:
                selected = tuple(range(num_layers))

            teacher_cpu = {
                layer_id: teacher[layer_id + 1].detach().cpu()
                for layer_id in selected
            }
            student_map = {
                layer_id: student_blocks[layer_id].clone().detach().requires_grad_(True)
                for layer_id in selected
            }
            # Rebuild fresh leaf grads for selected comparison.
            for block in student_blocks:
                if block.grad is not None:
                    block.grad = None

            selected_student = {
                layer_id: tensor.clone().detach().requires_grad_(True)
                for layer_id, tensor in student_map.items()
            }
            selected_loss = compute_selected_distill_hidden_alignment_loss(
                teacher_hidden_by_layer=teacher_cpu,
                student_hidden_by_layer=selected_student,
                hidden_layer_indices=selected,
                attention_mask=attention_mask,
                layer_weighting=layer_weighting,
                num_layers=num_layers,
                loss_device=selected_student[selected[0]].device,
            )
            self.assertTrue(
                torch.allclose(selected_loss, reference.detach(), atol=1e-5, rtol=1e-5)
            )
            selected_loss.backward()

            for layer_id in selected:
                self.assertTrue(
                    torch.allclose(
                        selected_student[layer_id].grad,
                        reference_grads[layer_id],
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )


if __name__ == "__main__":
    unittest.main()
