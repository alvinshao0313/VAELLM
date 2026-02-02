import torch
import torch.nn as nn
import torch.nn.functional as F

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def get_disc_loss(disc_loss_type):
    if disc_loss_type == 'vanilla':
        disc_loss = vanilla_d_loss
    elif disc_loss_type == 'hinge':
        disc_loss = hinge_d_loss
    return disc_loss

def adopt_weight(global_step, threshold=0, value=0., warmup=0):
    if global_step < threshold or threshold < 0:
        weight = value
    else:
        weight = 1
        if global_step - threshold < warmup:
            weight = min((global_step - threshold) / warmup, 1)
    return weight