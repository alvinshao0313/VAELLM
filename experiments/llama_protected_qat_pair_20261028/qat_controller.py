"""Actual Llama original-coordinate protection STE and native uint8 transitions."""
from contextlib import contextmanager
from types import MethodType
import qat_common as c
from qat_state import optimization_proof

torch = c.torch
math_source = c.R / 'qwen_protected_native_probe/adapter.py'
math = c.native.module(math_source, '_llama_qat_readonly_original_q_math')


class QController:
    _coordinates = math.ProtectionController._coordinates
    _rows = math.ProtectionController._rows

    def __init__(self, model, modules, record):
        self.model, self.mode, self.entries, self.parameters = model, 'ste', {}, {}
        self.installed = False
        registered = {name: id(p) for name, p in model.named_parameters()}
        state_keys = set(model.state_dict())
        c.require(len(modules) == 224 and len(state_keys) == 3619, 'Require complete native Llama sparse005 topology.')
        for name, module in modules.items():
            q, scale, index = module.protected_input_qvalues, module.protected_input_scales, module.protected_input_indices
            declared = record['q_parameters'][name + '.protected_input_qvalues']
            c.require(module.protected_channel_quant_format == 'int8' and q.dtype == torch.uint8
                and q.ndim == 2 and q.is_contiguous() and q.device.type in ('cpu', 'cuda')
                and q.shape[1] == module.out_features and list(q.shape) == declared['shape']
                and c.tensor_record(q) == declared['source_tensor']
                and scale.dtype == torch.bfloat16 and scale.shape == (q.shape[0],) and scale.device == q.device
                and index.dtype == torch.int64 and index.shape == scale.shape and not q.requires_grad and not scale.requires_grad
                and module.protected_output_indices is None and module.trainable_decode and not module.cache_decoded_weight
                and module._cached_weight is None and '_decode_protected_input_weight_rows' not in module.__dict__
                and '_decode_weight' not in module.__dict__, 'Wrong actual Llama fixed-scale q layout: ' + name)
            parameter = torch.nn.Parameter(q.view(torch.int8).float())
            self.parameters[name + '.protected_code_coordinate'] = parameter
            self.entries[name] = {'module': module, 'source': q.detach().clone(), 'parameter': parameter,
                'previous_hard': q.detach().clone(), 'calls': 0}
            with torch.no_grad():
                original = module._decode_protected_input_weight_rows(dtype=torch.bfloat16, device=q.device)
                c.require(torch.equal(original, self._rows(name, dtype=torch.bfloat16, device=q.device)),
                    'Initial signed-q master must reproduce actual native protection exactly: ' + name)
        c.require(len(self.parameters) == 224 and sum(p.numel() for p in self.parameters.values()) == 113246208,
            'Wrong actual Llama protection inventory.')
        for name, entry in self.entries.items():
            def rows(instance, *, dtype, device, key=name):
                e = self.entries[key]
                c.require(instance is e['module'], 'Live q row adapter called on another module.')
                e['calls'] += 1
                return self._rows(key, dtype=dtype, device=device)
            entry['override'] = MethodType(rows, entry['module'])
            entry['module']._decode_protected_input_weight_rows = entry['override']
        self.installed = True
        c.require(set(model.state_dict()) == state_keys and {name: id(p) for name, p in model.named_parameters()} == registered,
            'External q masters must preserve native model registration.')

    def native_payload(self):
        return {name + '.protected_input_qvalues': self._coordinates(name)[0].detach().to(torch.int8).view(torch.uint8).contiguous()
            for name in self.entries}

    @torch.no_grad()
    def statistics(self, *, advance=False):
        payload, rows = self.native_payload(), {}
        for name, entry in self.entries.items():
            hard = payload[name + '.protected_input_qvalues']
            source = entry['source'].view(torch.int8).float()
            master = entry['parameter'] - source
            rows[name] = {'q_elements': hard.numel(), 'FP32_master_changed_elements': int(torch.count_nonzero(master)),
                'FP32_master_abs_delta_max': float(master.abs().max()),
                'changed_q_elements': int(torch.count_nonzero(hard != entry['source'])),
                'adjacent_hard_changed_elements': int(torch.count_nonzero(hard != entry['previous_hard'])),
                'live_row_calls': entry['calls']}
            if advance: entry['previous_hard'].copy_(hard)
        totals = {key: sum(row[key] for row in rows.values()) for key in
            ('q_elements', 'FP32_master_changed_elements', 'changed_q_elements', 'adjacent_hard_changed_elements')}
        totals['FP32_master_abs_delta_max'] = max(row['FP32_master_abs_delta_max'] for row in rows.values())
        return {'totals': totals, 'modules': rows, 'hard_change_required_for_short2': False}

    @contextmanager
    def native_generation(self, active, optimizer, evidence):
        c.require(self.installed and all(p.grad is None for p in self.parameters.values()), 'Generate before gradient accumulation.')
        before = optimization_proof(active, optimizer)
        identities = {name: (id(e['module'].protected_input_qvalues), e['module'].protected_input_qvalues.data_ptr())
            for name, e in self.entries.items()}
        payload = self.native_payload()
        try:
            with torch.no_grad():
                for name, entry in self.entries.items():
                    module = entry['module']
                    c.require(module._decode_protected_input_weight_rows is entry['override'], 'Live q row adapter changed.')
                    module.protected_input_qvalues.copy_(payload[name + '.protected_input_qvalues'])
                    del module._decode_protected_input_weight_rows
            evidence.update(native_q_tensors=224, native_q_elements=113246208, native_class_rows_used=True,
                hard_payload_sha256=c.digest({key: c.tensor_record(value) for key, value in payload.items()}))
            yield
        finally:
            with torch.no_grad():
                for entry in self.entries.values():
                    module = entry['module']; module.clear_decoded_weight_cache()
                    module.protected_input_qvalues.copy_(entry['source'])
                    module._decode_protected_input_weight_rows = entry['override']
            c.require(before == optimization_proof(active, optimizer) and all(
                identities[name] == (id(e['module'].protected_input_qvalues), e['module'].protected_input_qvalues.data_ptr())
                and torch.equal(e['module'].protected_input_qvalues, e['source'])
                and e['module']._decode_protected_input_weight_rows is e['override']
                and e['module']._cached_weight is None for name, e in self.entries.items()),
                'Native OPD must restore exact D/q master, gradients, Adam, original buffers and live adapter.')
            evidence.update(FP32_D_q_Adam_exact=True, source_q_buffers_exact=True,
                initialized_leaves=before['initialized_leaves'], optimizer_steps=before['steps'])

    @torch.no_grad()
    def commit_terminal(self, active, optimizer):
        c.require(self.installed, 'Commit the actual endpoint once.')
        before = optimization_proof(active, optimizer); payload = self.native_payload()
        for name, entry in self.entries.items():
            module = entry['module']
            expected = self._rows(name, dtype=torch.bfloat16, device=entry['parameter'].device)
            module.protected_input_qvalues.copy_(payload[name + '.protected_input_qvalues'])
            del module._decode_protected_input_weight_rows
            c.require(torch.equal(expected, module._decode_protected_input_weight_rows(dtype=torch.bfloat16, device=expected.device)),
                'Committed native UINT8 q rows differ from hard STE rows.')
            module.clear_decoded_weight_cache()
        c.require(before == optimization_proof(active, optimizer), 'Commit must preserve actual masters/Adam.')
        self.installed = False
        return {'native_q_tensors': 224, 'native_rows_exact': True, 'instance_adapters_removed': True,
            'FP32_master_and_Adam_exact': True, 'scale_and_indices_written': False}
