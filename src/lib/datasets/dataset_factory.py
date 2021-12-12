from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset, MultiScaleJD
from .dataset.bdd100k import BDD100K
from .yolomot import YOLOMOT


def get_dataset(task, mult_scale=False):
    if task == 'mot':
        if mult_scale:
            return MultiScaleJD
        else:
            return YOLOMOT
        return JointDataset
    else:
        return None
