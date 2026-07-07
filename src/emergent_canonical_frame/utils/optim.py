import math

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def construct_optimizer(model: torch.nn.Module, total_iters, cfg: DictConfig):
    params = split_parameters(model, cfg.weight_decay, cfg.lr)

    if cfg.name == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    elif cfg.name == "Adam":
        optimizer = torch.optim.Adam(params, lr=cfg.lr)
    elif cfg.name == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg.lr)
    else:
        raise NotImplementedError("Optimizer not implemented")


    def make_lr_lambda(eta_min_ratio):
        def lr_lambda(step):
            if step >= total_iters - cfg.warmup_iters:
                return eta_min_ratio
            progress = step / (total_iters - cfg.warmup_iters)
            return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return lr_lambda

    lr_lambdas = []
    lr_lambdas.append(make_lr_lambda(cfg.eta_min / cfg.lr))

    cosine_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)

    return optimizer, cosine_scheduler


def split_parameters(model: torch.nn.Module, wd: float, lr: float):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.MultiheadAttention, nn.Parameter)
    blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn
            if not p.requires_grad:
                continue
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn in ("cls_token", "storage_tokens", "mask_token", "logit_scale", "mask_pos") or pn.endswith(".gamma"):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, (
        "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)
    )


    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": wd, "lr": lr},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0, "lr": lr},
    ]
    optim_groups = [g for g in optim_groups if len(g["params"]) > 0]
    return optim_groups
