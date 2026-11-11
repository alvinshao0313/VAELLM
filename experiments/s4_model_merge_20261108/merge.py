"""Interpolate aligned masters, never packed BSQ bytes or optimizer states."""
from . import bindings as b
from experiments.s4_scope_objective_20261107.protection import integer_coordinate


def blend(left,right,ratio):
    b.require(left.dtype==right.dtype==b.torch.float32 and left.shape==right.shape and 0<=ratio<=1,
        'Aligned FP32 masters and global ratio in[0,1] required.')
    if ratio==0:return left.clone()
    if ratio==1:return right.clone()
    value=(1-ratio)*left+ratio*right
    b.require(b.torch.isfinite(value).all(),'Nonfinite merged master.')
    return value


def native(name,value):
    if '._parallel_stage_decoder.' in name:return name,value.bfloat16()
    b.require(name.endswith('.protected_code_coordinate'),'Unexpected merge master: '+name)
    return name.rsplit('.',1)[0]+'.protected_input_qvalues',integer_coordinate(value).to(b.torch.int8).view(b.torch.uint8)


def active_keys(names):
    return {name if '._parallel_stage_decoder.' in name else name.rsplit('.',1)[0]+'.protected_input_qvalues' for name in names}
