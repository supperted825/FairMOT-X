from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

# from .networks.dlav0 import get_pose_net as get_dlav0
# from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
# from .networks.pose_hrnet import get_pose_net as get_pose_net_hrnet
# from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
# from .networks.resnet_fpn_dcn import get_pose_net as get_pose_net_fpn_dcn
# from .networks.csp_darknet import get_csp_darknet
# from .networks.regnet.regnet import RegNet0, RegNet
# from .networks.efficientdet import EfficientDet
# from .networks.yolonet import YOLONET

from .networks.yoloX import YOLOXMOT

_model_factory = {
    # 'dlav0': get_dlav0,  # default DLAup
    # 'dla': get_dla_dcn,
    # 'resdcn': get_pose_net_dcn,
    # 'resfpndcn': get_pose_net_fpn_dcn,
    # 'hrnet': get_pose_net_hrnet,
    # 'cspdarknet': get_csp_darknet,
    # 'regnet0': RegNet0,
    # 'regnet': RegNet,
    # 'effdet': EfficientDet,
    # 'yolonet': YOLONET
    'yolox' : YOLOXMOT
}


def create_model(arch, heads=None, head_conv=None, opt=None):
    """
    :param arch:
    :param heads:
    :param head_conv:
    :return:
    """
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0  # 模型架构
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    if arch in ['regnet0', 'regnet', 'effdet', 'yolonet']:
        model = get_model(heads=heads, head_convs=head_conv, opt=opt)
    elif arch in ['yolox']:
        model = get_model(opt=opt)
    else:
        model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)

    return model


def load_model(model,
               model_path,
               optimizer=None,
               resume=False,
               lr=None,
               lr_step=None):
    print(f"Loading Model from {model_path}")
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    if 'epoch' in checkpoint.keys():
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    if 'state_dict' in checkpoint.keys():
        state_dict_ = checkpoint['state_dict']
    else:
        state_dict_ = checkpoint
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
            
    if "yolox_" in model_path:
      state_dict = state_dict_["model"]
    
    model_state_dict = model.state_dict()
    
    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
            pass
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    """
    :param path:
    :param epoch:
    :param model:
    :param optimizer:
    :return:
    """
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    data = {'epoch': epoch,
            'state_dict': state_dict}

    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()

    torch.save(data, path)
