import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from bitvae.utils.misc import is_torch_optimizer

def load_unstrictly(state_dict, model, loaded_keys=[]):
    missing_keys = []
    for name, param in model.named_parameters():
        if name in state_dict:
            try:
                param.data.copy_(state_dict[name])
            except:
                # print(f"{name} mismatch: param {name}, shape {param.data.shape}, state_dict shape {state_dict[name].shape}")
                missing_keys.append(name)
        elif name not in loaded_keys:
            missing_keys.append(name)
    return model, missing_keys

def resume_from_ckpt(state_dict, model_optims, load_optimizer=True):
    all_missing_keys = []
    # load weights first
    for k in model_optims:
        if model_optims[k] and (not is_torch_optimizer(model_optims[k])) and k in state_dict:
            model_optims[k], missing_keys = load_unstrictly(state_dict[k], model_optims[k])
            all_missing_keys += missing_keys
        
    if len(all_missing_keys) == 0 and load_optimizer:
        print("Loading optimizer states")
        for k in model_optims: 
            if model_optims[k] and is_torch_optimizer(model_optims[k]) and k in state_dict:
                model_optims[k].load_state_dict(state_dict[k])
    else:
        print(f"missing weights: {all_missing_keys}, do not load optimzer states")
    return model_optims, state_dict["step"]
