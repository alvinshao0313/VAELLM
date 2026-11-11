"""Run the mature real native gate fit, replacing only VAE optimizer factories."""
import argparse
from pathlib import Path

from . import bindings as b
from .optimizer import install


def execute(phase):
    from experiments.cat_geometry_cohort_cmcc_20261102 import fit as original
    from train_utils import cat_train_pipeline as native
    from .evaluate import deployed_logits, warm

    b.prepared()
    short, records = phase == 'short', []
    save = native.save_cat_v6_full_checkpoint

    def capture(*args, **kwargs):
        result = save(*args, **kwargs)
        if kwargs['checkpoint_kind'] == 'round_base' and kwargs.get('category') == 'gate_proj':
            model, directory = args[0], Path(result['output_dir'])
            model.eval().requires_grad_(False)
            prepared = warm(model, 180)
            runtime = b.torch.load(Path(b.base.sources()['after_o']['path']) / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
            tokens = runtime['activation_runtime']['cache']['input_ids'][0].reshape(1,256).clone()
            expected = deployed_logits(model, tokens)
            meta = b.read(directory / 'checkpoint_meta.json')
            state = b.torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
            digest = b.base.fork.state_digest(state)
            live = {name:value.detach().cpu() for name,value in model.state_dict().items()}
            b.require(b.base.fork.state_digest(live) == digest, 'Deployment prewarm changed a persistent saved field.')
            b.torch.save(dict(checkpoint_id=b.read(directory / 'checkpoint_meta.json')['checkpoint_id'],
                full_state=digest, input_ids=tokens, logits=expected, warmup=prepared),
                b.OUTPUT / 'muon' / phase / 'same_input_deployment.pt')
        return result

    native.save_cat_v6_full_checkpoint = capture
    try:
        with install(native, records, short=short):
            original.execute('muon', phase)
    finally:
        native.save_cat_v6_full_checkpoint = save
    path = b.OUTPUT / 'muon' / phase / 'report.json'
    report = b.read(path)
    b.require(report['status'] == 'raw2_completed' and report['exit_code'] == 0 and len(records) == 2
        and all(row['parameter_elements'] == 1211904 and row['calibration']['relative_error'] <= .01
            and len(row['updates']) == (2 if short else 3) for row in records),
        'Both actual gate stages must own the full BF16 matrices, independently calibrate and execute the fixed updates.')
    report.update(contract=b.CONTRACT, optimizer_stages=records,
        actual_optimizer='Mature BF16 per-linear VAE Muon; original BF16 auxiliary AdamW',
        historical_control_init_digest_available=False, same_input_evidence=str(path.parent / 'same_input_deployment.pt'))
    b.write(path, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('short','formal'))
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    execute(parser.parse_args().phase)
