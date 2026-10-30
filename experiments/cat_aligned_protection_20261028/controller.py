"""Original scale/INT8 STE mathematics with source-derived protection geometry."""
from contextlib import contextmanager
from types import MethodType
import bindings as b
from snapshot import optimization_proof


def effective_scales(source, log_delta):
    value = source.float() * log_delta.exp()
    scales = value.to(b.torch.bfloat16)
    b.require(log_delta.dtype == b.torch.float32 and b.torch.isfinite(value).all().item()
        and (value > 0).all().item() and b.torch.isfinite(scales).all().item() and (scales > 0).all().item(),
        'Learned scales must be finite/positive before and after native BF16 rounding.')
    return scales


def signed_coordinates(master):
    b.require(master.dtype == b.torch.float32 and b.torch.isfinite(master).all().item(), 'Require finite FP32 signed-q master.')
    clipped = master.clamp(-128., 127.)
    return clipped + (clipped.round() - clipped).detach()


def scale_rows(q, scale, delta, *, dtype):
    return (q.view(b.torch.int8).float() * effective_scales(scale, delta).float().unsqueeze(1)).to(dtype).contiguous()


def q_rows(master, scale, *, dtype):
    return (signed_coordinates(master) * scale.float().unsqueeze(1)).to(dtype).contiguous()


class ProtectionController:
    def __init__(self, model, modules, record):
        self.model, self.kind = model, record['protection_kind']
        self.entries, self.parameters, self.installed = {}, {}, False
        registered = {name: id(p) for name, p in model.named_parameters()}
        state_keys = set(model.state_dict())
        b.require(self.kind in ('scale', 'q') and len(modules) == 252 and len(state_keys) == 3423,
            'Require all252 native live decoders.')
        self.suffix = '.protected_input_scales' if self.kind == 'scale' else '.protected_input_qvalues'
        for name, module in modules.items():
            q, scale, index = module.protected_input_qvalues, module.protected_input_scales, module.protected_input_indices
            declared = record['protection_parameters'][name + self.suffix]
            source = scale if self.kind == 'scale' else q
            b.require(module.protected_channel_quant_format == 'int8' and q.dtype == b.torch.uint8 and q.ndim == 2
                and q.is_contiguous() and q.device.type == 'cuda' and q.shape[1] == module.out_features
                and scale.dtype == b.torch.bfloat16 and scale.shape == (q.shape[0],) and scale.device == q.device
                and index.dtype == b.torch.int64 and index.shape == scale.shape
                and list(source.shape) == declared['shape'] and b.core.tensor_record(source) == declared['source_tensor']
                and not q.requires_grad and not scale.requires_grad and module.protected_output_indices is None
                and module.trainable_decode and not module.cache_decoded_weight and module._cached_weight is None
                and '_decode_protected_input_weight_rows' not in module.__dict__ and '_decode_weight' not in module.__dict__,
                'Changed original native live protection layout: ' + name)
            master = b.torch.nn.Parameter(b.torch.zeros_like(scale, dtype=b.torch.float32)
                if self.kind == 'scale' else q.view(b.torch.int8).float())
            key = name + ('.protected_log_scale_delta' if self.kind == 'scale' else '.protected_code_coordinate')
            self.parameters[key] = master
            self.entries[name] = dict(module=module, source=source.detach().clone(), parameter=master,
                previous_hard=q.detach().clone() if self.kind == 'q' else None)
            with b.torch.no_grad():
                b.require(b.torch.equal(module._decode_protected_input_weight_rows(dtype=b.torch.bfloat16, device=q.device),
                    self.rows(name, dtype=b.torch.bfloat16, device=q.device)), 'Initial learned rows must exactly match native rows.')
        b.require(len(self.parameters) == 252 and sum(p.numel() for p in self.parameters.values()) == record['protection_elements'],
            'External protection masters differ from actual source geometry.')
        for name, entry in self.entries.items():
            def rows(instance, *, dtype, device, key=name):
                b.require(instance is self.entries[key]['module'], 'Wrong protection-row override owner.')
                return self.rows(key, dtype=dtype, device=device)
            entry['override'] = MethodType(rows, entry['module'])
            entry['module']._decode_protected_input_weight_rows = entry['override']
        self.installed = True
        b.require(set(model.state_dict()) == state_keys and {name: id(p) for name, p in model.named_parameters()} == registered,
            'External masters must not alter native state or parameter registration.')

    def rows(self, name, *, dtype, device):
        entry = self.entries[name]
        b.require(b.torch.device(device) == entry['parameter'].device, 'Protection master/native row device mismatch.')
        module = entry['module']
        if self.kind == 'scale':
            return scale_rows(module.protected_input_qvalues, entry['source'], entry['parameter'], dtype=dtype)
        return q_rows(entry['parameter'], module.protected_input_scales, dtype=dtype)

    def native_payload(self):
        return {name + self.suffix: (effective_scales(e['source'], e['parameter']).detach().clone()
            if self.kind == 'scale' else signed_coordinates(e['parameter']).detach().to(b.torch.int8).view(b.torch.uint8).contiguous())
            for name, e in self.entries.items()}

    def buffer_proof(self):
        return {name: dict(identity=id(getattr(e['module'], self.suffix[1:])),
            data_ptr=getattr(e['module'], self.suffix[1:]).data_ptr(),
            value=b.core.tensor_record(getattr(e['module'], self.suffix[1:]))) for name, e in self.entries.items()}

    @b.torch.no_grad()
    def statistics(self, *, advance=False):
        native = self.native_payload()
        if self.kind == 'scale':
            return dict(FP32_log_changed_elements=sum(int((e['parameter'] != 0).sum()) for e in self.entries.values()),
                BF16_scale_changed_elements=sum(int((native[n + self.suffix] != e['source']).sum()) for n, e in self.entries.items()),
                log_abs_max=max(float(e['parameter'].abs().max()) for e in self.entries.values()),
                scale_relative_abs_max=max(float((e['parameter'].exp()-1).abs().max()) for e in self.entries.values()))
        rows = {}
        for name, e in self.entries.items():
            hard = native[name + self.suffix]
            delta = hard.view(b.torch.int8).float() - e['source'].view(b.torch.int8).float()
            adjacent = hard.view(b.torch.int8).float() - e['previous_hard'].view(b.torch.int8).float()
            master_delta = e['parameter'] - e['source'].view(b.torch.int8).float()
            rows[name] = dict(q_elements=hard.numel(), FP32_master_changed_elements=int(b.torch.count_nonzero(master_delta)),
                FP32_master_abs_delta_max=float(master_delta.abs().max()), changed_q_elements=int(b.torch.count_nonzero(delta)),
                adjacent_hard_changed_elements=int(b.torch.count_nonzero(adjacent)), code_abs_delta_max=float(delta.abs().max()),
                master_outside_int8_range=int(((e['parameter'] < -128) | (e['parameter'] > 127)).sum()))
            if advance:
                e['previous_hard'].copy_(hard)
        sums = ('q_elements', 'FP32_master_changed_elements', 'changed_q_elements', 'adjacent_hard_changed_elements', 'master_outside_int8_range')
        totals = {key: sum(row[key] for row in rows.values()) for key in sums}
        totals.update(FP32_master_abs_delta_max=max(r['FP32_master_abs_delta_max'] for r in rows.values()),
            code_abs_delta_max=max(r['code_abs_delta_max'] for r in rows.values()))
        return dict(totals=totals, modules=rows, hard_change_required_for_short2=False)

    @contextmanager
    def native_generation(self, decoder, optimizer, evidence):
        b.require(self.installed and all(p.grad is None for p in self.parameters.values()), 'Generate before gradient accumulation.')
        before = optimization_proof(decoder, self.parameters, optimizer)
        buffers, native = self.buffer_proof(), self.native_payload()
        try:
            with b.torch.no_grad():
                for name, e in self.entries.items():
                    module = e['module']
                    b.require(module._decode_protected_input_weight_rows is e['override'], 'Protection row override changed.')
                    getattr(module, self.suffix[1:]).copy_(native[name + self.suffix])
                    del module._decode_protected_input_weight_rows
                    b.require(b.torch.equal(getattr(module, self.suffix[1:]), native[name + self.suffix]), 'Native hard payload install differs.')
            evidence.update(kind=self.kind, native_tensors=252, native_elements=sum(v.numel() for v in native.values()),
                native_payload_sha256=b.core.mapping_digest({k: b.core.tensor_record(v) for k, v in native.items()}),
                original_native_class_rows_used=True)
            yield
        finally:
            with b.torch.no_grad():
                for e in self.entries.values():
                    module = e['module']
                    module.clear_decoded_weight_cache()
                    getattr(module, self.suffix[1:]).copy_(e['source'])
                    module._decode_protected_input_weight_rows = e['override']
            after = optimization_proof(decoder, self.parameters, optimizer)
            b.require(before == after and buffers == self.buffer_proof() and all(e['module']._cached_weight is None
                and e['module']._decode_protected_input_weight_rows is e['override'] for e in self.entries.values()),
                'Generation altered FP32 master/Parameter/gradient/Adam or failed native protection/cache restoration.')
            evidence.update(before=before, after=after, FP32_D_and_protection_Adam_exact=True,
                original_native_buffers_restored_exact=True)

    @b.torch.no_grad()
    def commit_terminal(self, decoder, optimizer):
        b.require(self.installed, 'Commit only the genuine terminal native payload.')
        before, native = optimization_proof(decoder, self.parameters, optimizer), self.native_payload()
        for name, e in self.entries.items():
            module = e['module']
            expected = self.rows(name, dtype=b.torch.bfloat16, device=e['parameter'].device)
            getattr(module, self.suffix[1:]).copy_(native[name + self.suffix])
            del module._decode_protected_input_weight_rows
            b.require(b.torch.equal(expected, module._decode_protected_input_weight_rows(dtype=b.torch.bfloat16, device=expected.device)),
                'Terminal native deployment rows differ from differentiable hard rows.')
            module.clear_decoded_weight_cache()
        b.require(before == optimization_proof(decoder, self.parameters, optimizer), 'Terminal commit changed continuous masters/Adam.')
        self.installed = False
        return dict(kind=self.kind, native_tensors=252, native_rows_exact=True, instance_overrides_removed=True,
            FP32_master_and_Adam_unchanged=True)
