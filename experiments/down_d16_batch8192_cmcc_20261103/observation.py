"""Observe the real factory and first training inputs without consuming RNG."""
from contextlib import contextmanager

from . import bindings as b
from experiments.adaptive_class_20261030 import initialization


@contextmanager
def observe(native, output):
    import litebsq.llm_vae as llm
    factory, create = llm.MultiLayerVAE, native.create_optimizer
    expected = b.initialization_reference()
    inputs, models = [], []
    with initialization.observe(native) as rows:
        capture = native.create_optimizer

        def build(args):
            model = factory(args)
            index = len(models)
            models.append(model)
            row = dict(actual_stage_index=index, training_forward_calls=0)
            inputs.append(row)

            def first(_model, args, kwargs):
                if not kwargs.get('is_train', True):
                    return
                row['training_forward_calls'] += 1
                if row['training_forward_calls'] != 1:
                    return
                value = args[0]
                b.require(list(value.shape) == [8192, 36, 16]
                    and value.dtype == b.torch.bfloat16,
                    'The actual first train input must be BF16 [8192,36,16].')
                row.update(first_training_input_shape=list(value.shape), input_dtype=str(value.dtype),
                    batch_per_layer=8192, scalar_exposure_per_layer_update=131072)
                b.write(output / 'effective_factory.json', dict(initializations=rows, inputs=inputs))
                print('ACTUAL_TRAIN_INPUT', row, flush=True)

            model.register_forward_pre_hook(first, with_kwargs=True)
            return model

        def optimizer(*args, **kwargs):
            result = capture(*args, **kwargs)
            index = len(rows) - 1
            b.require(index < 2 and all(rows[index][key] == expected[index][key]
                for key in ('initial_parameter_digest', 'parameter_elements', 'parameter_dtypes',
                    'parameter_shapes', 'initial_learning_rates')),
                'Actual factory initialization differs from the matching original D16 stage.')
            b.require(type(result) is b.torch.optim.AdamW and all(group['betas'] == (.9, .95)
                and group['eps'] == 1e-8 and group['weight_decay'] == 0 for group in result.param_groups),
                'Keep the original AdamW beta/eps/WD contract.')
            rows[index]['matching_original_D16_stage'] = True
            b.write(output / 'effective_factory.json', dict(initializations=rows, inputs=inputs))
            print('ACTUAL_FACTORY_INITIALIZATION', rows[index], flush=True)
            return result

        llm.MultiLayerVAE, native.create_optimizer = build, optimizer
        try:
            yield rows, inputs
            b.require(len(rows) == len(inputs) == 2 and all(row['training_forward_calls'] == 10000
                and row['batch_per_layer'] == 8192 for row in inputs),
                'Require both real stages with 10000 B8192 training forwards.')
            b.write(output / 'effective_factory.json', dict(initializations=rows, inputs=inputs))
        finally:
            llm.MultiLayerVAE, native.create_optimizer = factory, create
