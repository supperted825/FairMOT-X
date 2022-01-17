#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn
from .yolox.yolo_head import YOLOXHead
from .yolox.yolo_pafpn import YOLOPAFPN

class YOLOXMOT(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    
    Modified for FairMOT Head Outputs & Loss Calculations
    """

    def __init__(self, backbone=None, head=None, opt=None):
        super().__init__()

        self.backbone = YOLOPAFPN(opt.yolo_depth, opt.yolo_width)
        self.head = YOLOXHead(opt.num_classes, width=opt.yolo_width, opt=opt)

    def forward(self, x, targets=None):
        
        # FPN outputs content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        
        if self.training:
            
            # Targets Must be Passed for Loss Calculation
            assert targets is not None
            
            # Get Losses After Passing Through Head
            loss, iou_loss, conf_loss, cls_loss, l1_loss, reid_loss, num_fg = self.head(fpn_outs, targets, x)
            
            outputs = loss, {
                "tot_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "reid_loss": reid_loss,
                "num_fg": num_fg
            }
            
        else:
            outputs = self.head(fpn_outs)

        return outputs
