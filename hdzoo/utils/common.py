"""
HD Zoo - Yeseong Kim (CELL) @ DGIST, 2023
"""

import torch
import numpy as np
import random

from ..utils.logger import log

""" Initialize the random seed"""
def setup_seed(seed):
    # Random seed initialization
    log.d("Random seed: {}".format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


""" Convert a list of np array to tensors and allocate to cuda if available """
def to_tensor(*X):
    ret = []
    for x in X:
        x = torch.from_numpy(x).float()
        if torch.cuda.is_available():
            x = x.cuda()
        ret.append(x)
    return ret
