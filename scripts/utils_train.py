import numpy as np
import os
import lib
from mldm.modules import MLPDiffusion, ResNetDiffusion

def get_model(
    model_name,
    model_params,
): 

    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


def make_dataset(
    data_path: str,
    T: lib.Transformations,
    strategy: str,
    pct: int
):

    D = lib.Dataset.from_dir(data_path, strategy, pct)
    newD = []
    for d in D:
        newD.append(lib.transform_dataset(d, T))

    return newD