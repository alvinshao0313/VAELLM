"""Pin reused PUBLIC components and this isolated implementation."""
from pathlib import Path

from experiments.cat_auto_protocol_cmcc_20261104.live import source_pins as native_pins, _sha


def source_pins():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [Path('experiments/sc_cat_reproducibility_cmcc_20261109/execution.py').resolve()]
    paths += list(Path(__file__).parent.glob('*.sh'))
    paths += [Path('experiments/muon_compact_cmcc_20261030/optimizer.py').resolve()]
    for package, names in {
        'sc_cat_live_cmcc_20261106': ('bindings','live_io','fit_worker','recovery_worker','evaluate'),
        'sc_cat_finish_cmcc_20261108': ('resume',),
        'sc_cat_search_cmcc_20261105': ('evidence',),
        'sc_cat_continue_cmcc_20261107': ('longbench','final_score'),
        'cat_offline_execution_cmcc_20261104': ('protocol','evaluate'),
    }.items():
        paths += [Path('experiments',package,name+'.py').resolve() for name in names]
    paths += list(Path('experiments/longbench_v1').glob('*.py'))
    paths += [Path('experiments/compare_longbench_v1.py')]
    return dict(native_pins(), **{str(p.resolve()): _sha(p) for p in paths})
