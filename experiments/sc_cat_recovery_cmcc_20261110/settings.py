"""The confirmed recovery search space; no Cartesian search across rounds."""
from dataclasses import asdict, dataclass, replace
from itertools import product

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require


ROUNDS = ('combination', 'scope', 'rank', 'alpha', 'dropout', 'optimizer', 'decoder_lr', 'protection_lr', 'lora_lr')
CALIBRATED_ROUNDS = tuple(name for name in ROUNDS if name != 'dropout')


@dataclass(frozen=True)
class RecoverySettings:
    groups: str = 'DL'
    decoder_scope: str = 'prefix'
    protection_scope: str = 'prefix'
    decoder_lr: float = 1e-5
    protection_multiplier: float = 1.
    lora_lr: float = 1e-4
    lora_rank: int = 12
    lora_alpha: int = 24
    lora_dropout: float = .1
    bound_optimizer: str = 'adamw'
    lora_optimizer: str = 'adamw'
    schedule_max: int = 500

    def __post_init__(self):
        require(self.groups in ('D', 'P', 'L', 'DP', 'DL', 'PL', 'DPL'), 'Use a nonempty confirmed D/P/L combination.')
        require(self.decoder_scope in ('current', 'prefix') and self.protection_scope in ('current', 'prefix'), 'Invalid category range.')
        require(self.decoder_lr in (5e-6, 1e-5, 2e-5) and self.lora_lr in (5e-5, 1e-4, 2e-4)
            and self.protection_multiplier in (.5, 1., 2.), 'Outside the confirmed LR grid.')
        require(self.lora_rank in (6, 12, 24, 48) and self.lora_alpha in (self.lora_rank, 2*self.lora_rank, 4*self.lora_rank)
            and self.lora_dropout in (0., .1), 'Outside the confirmed LoRA grid.')
        require(self.bound_optimizer in ('adamw', 'muon') and self.lora_optimizer in ('adamw', 'muon')
            and self.schedule_max in (500, 1000), 'Invalid optimizer or schedule.')

    @property
    def scope(self):
        # Native metadata mode remains a valid existing enum; the immutable
        # contract additionally binds the full independent D/P/L settings.
        if 'L' in self.groups:
            return 'remaining_lora_prefix_decoder' if self.decoder_scope == 'prefix' else 'remaining_lora_current_decoder'
        return 'current_decoder'

    @property
    def node_steps(self):
        return tuple(range(125, self.schedule_max+1, 125))


def normalize(settings, *, prefix, current, protected, future):
    """Deduplicate by actual owner inventories, including absent P/L and first class."""
    prefix, current, protected, future = map(tuple, (prefix, current, protected, future))
    require(current and set(current) <= set(prefix) and not set(prefix) & set(future), 'Invalid live inventories.')
    d = (prefix if settings.decoder_scope == 'prefix' else current) if 'D' in settings.groups else ()
    p = tuple(n for n in (prefix if settings.protection_scope == 'prefix' else current) if n in protected) if 'P' in settings.groups else ()
    l = future if 'L' in settings.groups else ()
    groups = ''.join(g for g, names in (('D', d), ('P', p), ('L', l)) if names)
    if not groups:
        return None
    values = dict(groups=groups,
        decoder_scope='prefix' if not d or set(d) == set(prefix) else 'current',
        protection_scope='prefix' if not p or set(p) == set(prefix) & set(protected) else 'current')
    if not d:
        values['decoder_lr'] = 1e-5
    if not p:
        values['protection_multiplier'] = 1.
    if not l:
        values.update(lora_rank=12, lora_alpha=24, lora_dropout=.1, lora_lr=1e-4, lora_optimizer='adamw')
    if not d and not p:
        values['bound_optimizer'] = 'adamw'
    return replace(settings, **values)


def round_candidates(name, base, **inventory):
    require(name in ROUNDS and base.schedule_max == 500, 'Generate search rounds on the original500 schedule.')
    values = []
    if name == 'combination':
        values = [replace(base, groups=g, decoder_scope='prefix', protection_scope='prefix')
            for g in ('D', 'P', 'L', 'DP', 'DL', 'PL', 'DPL')]
    elif name == 'scope':
        values = [replace(base, decoder_scope=d, protection_scope=p) for d, p in product(
            ('current', 'prefix') if 'D' in base.groups else ('prefix',),
            ('current', 'prefix') if 'P' in base.groups else ('prefix',))]
    elif name == 'rank' and 'L' in base.groups:
        values = [replace(base, lora_rank=r, lora_alpha=2*r) for r in (6, 12, 24, 48)]
    elif name == 'alpha' and 'L' in base.groups:
        values = [replace(base, lora_alpha=m*base.lora_rank) for m in (1, 2, 4)]
    elif name == 'dropout' and 'L' in base.groups:
        values = [replace(base, lora_dropout=p) for p in (0., .1)]
    elif name == 'optimizer':
        values = [replace(base, bound_optimizer=d, lora_optimizer=l) for d, l in product(
            ('adamw', 'muon') if set(base.groups) & set('DP') else ('adamw',),
            ('adamw', 'muon') if 'L' in base.groups else ('adamw',))]
    elif name == 'decoder_lr' and 'D' in base.groups:
        values = [replace(base, decoder_lr=x) for x in (5e-6, 1e-5, 2e-5)]
    elif name == 'protection_lr' and 'P' in base.groups:
        values = [replace(base, protection_multiplier=x) for x in (.5, 1., 2.)]
    elif name == 'lora_lr' and 'L' in base.groups:
        values = [replace(base, lora_lr=x) for x in (5e-5, 1e-4, 2e-4)]
    unique = {}
    for value in values:
        value = normalize(value, **inventory)
        if value is not None:
            unique[digest(asdict(value))] = value
    return tuple(unique.values())


def trajectory_id(raw_id, settings, data_identity, code_identity):
    return digest(dict(raw_id=raw_id, settings=asdict(settings), data_identity=data_identity, code_identity=code_identity))
