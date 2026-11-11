"""Original explicit D16/B16384 live fitter with only its two optimizer factories replaced."""
from . import bindings as b
from . import native_fit
from .optimizer import install


def execute(model,parsed,activation,plan,*,output,short,records):
    prepared=b.read(b.PREPARED)
    with install(b.recovery.native,records,short=short):
        report=native_fit.execute(model,parsed,activation,plan,arm='d16_muon',output=output,short=short)
    expected=prepared['controls']['D16_initializations']
    matches=[actual['canonical_original_flat_order_initial_digest']==old['initial_parameter_digest']
        for actual,old in zip(records,expected,strict=True)]
    b.require(len(records)==2 and all(matches) and report['vector_batch_per_layer']==16384
        and all(row['calibration']['relative_error']<=.01 and row['parameter_elements']==397728
            and len(row['updates'])==(2 if short else 3) for row in records),
        'Both actual D16 stage flat initializations, batch/owners and freshly calibrated kappa must match this candidate contract.')
    report.update(actual_optimizer='Original BF16 VAE per-linear Muon math, D16 owners; original BF16 auxiliary AdamW',
        original_D16_flat_initialization_matches=matches,optimizer_stages=records,
        canonical_initialization_boundary='Compare original one-group flat order to the actual D16 Adam fit, not D32 or the new two-group observer order.',
        persistent_FP32_parameter_master=False)
    return report
