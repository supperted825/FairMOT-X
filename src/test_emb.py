from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2
import math

import pickle

from sklearn import metrics
from scipy import interpolate
import numpy as np
from tqdm import tqdm
from torchvision.transforms import transforms as T
import torch.nn.functional as F
from models.model import create_model, load_model
from lib.datasets.dataset_factory import get_dataset
from opts import opts


def test_emb(
        opt,
        batch_size=16,
        img_size=(1024, 576),
        print_interval=100, ):
    """
    :param opt:
    :param batch_size:
    :param img_size:
    :param print_interval:
    :return:
    """
    data_cfg = opt.data_cfg
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    nC = 1
    test_paths = list(data_cfg_dict['test_emb'].values())[0]
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
        
    print('Creating model...')
    model = create_model(opt.arch, opt=opt)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()
    
    Dataset = get_dataset(opt.task, opt.multi_scale)  # if opt.task==mot -> JointDataset

    # Get data loader
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(test_paths, opt=opt)
    
    print("Length of Dataset: ", len(dataset))
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=8, drop_last=False, collate_fn=dataset.collate_fn)
    
    emb_scale_dict = dict()
    for cls_id, nID in opt.nID_dict.items():
        emb_scale_dict[cls_id] = math.sqrt(2) * math.log(nID - 1)
    embedding, id_labels = [], []
    
    print('Extracting features...')
    for batch_i, (imgs, det_labels, track_ids) in enumerate(data_loader):
        
        id_head = []
        batch_id_labels = []
        
        imgs = imgs.float().to(device=opt.device, non_blocking=True)
        multibatch_det_labels = det_labels.to(device=opt.device, non_blocking=True)
        multibatch_track_ids = track_ids.to(device=opt.device, non_blocking=True)
        
        outputs, reid_features = model.forward(imgs)

        for batch_idx in range(outputs.shape[0]):
            num_gt = multibatch_det_labels[multibatch_det_labels[:, 0] == batch_idx].shape[0]
        
            batch_det_labels = multibatch_det_labels[multibatch_det_labels[:, 0] == batch_idx]
            gt_bboxes_per_image = batch_det_labels[ :num_gt, 2:6]

            # ----- ReID Loss Calculation
            
            # Nothing to Train if No Ground Truth IDs
            if num_gt == 0:
                continue
            
            # ReID Feature Map for this Image
            img_features = reid_features[batch_idx]       # Ch x H x W
            _, id_map_h, id_map_w = img_features.shape
            
            # Extract Center Coordinates of GT bboxes and Scale - center_xs, center_ys are arrays
            ny, nx = imgs[batch_idx].shape[1], imgs[batch_idx].shape[2]
            center_xs = gt_bboxes_per_image[:,0] * id_map_w / nx
            center_ys = gt_bboxes_per_image[:,1] * id_map_h / ny

            # Convert Center Coordinates to Int64
            center_xs += 0.5
            center_ys += 0.5
            center_xs = center_xs.long()
            center_ys = center_ys.long()

            # Clip to stay within ReID Feature Map Range
            center_xs.clamp_(0, id_map_w - 1)
            center_ys.clamp_(0, id_map_h - 1)
            
            # Extract ReID Feature at Center Coordinates
            # Since img_features has opt.reid_dim channels, we get 128 x nL, then transpose
            id_head.extend(img_features[..., center_ys, center_xs].T)
            
            batch_id_labels.extend(multibatch_track_ids[multibatch_track_ids[:, 0] == batch_idx][:, 1])
        
        for i in range(0, len(id_head)):
            if len(id_head[i].shape) == 0:
                continue
            else:
                feat, label = id_head[i], batch_id_labels[i]
            if label != -1:
                embedding.append(feat)
                id_labels.append(label)
        
        if batch_i % print_interval == 0:
            pass
            # print(f"Num Identities: {len(id_labels)}")
            
    embedding = torch.stack(embedding, dim=0).cuda().to(torch.float16)
    id_labels = torch.LongTensor(id_labels)
    n = len(id_labels)
    print(n, len(embedding))
    assert len(embedding) == n

    print("Preparing Data...")
    embedding = F.normalize(embedding, dim=1)
    p_dist = torch.mm(embedding, embedding.T).cpu().numpy()
    gt = id_labels.expand(n, n).eq(id_labels.expand(n, n).t()).numpy()

    
    print("Calculating Metrics...")
    up_triangle = np.where(np.triu(p_dist) - np.eye(n) * p_dist != 0)
    p_dist = p_dist[up_triangle]
    gt = gt[up_triangle]

    far_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    far, tar, threshold = metrics.roc_curve(gt, p_dist)
    interp = interpolate.interp1d(far, tar)
    tar_at_far = [interp(x) for x in far_levels]
    print(opt.load_model)
    for f, fa in enumerate(far_levels):
        print('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]), flush=True)
    return tar_at_far

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    with torch.no_grad():
        map = test_emb(opt, batch_size=16)
