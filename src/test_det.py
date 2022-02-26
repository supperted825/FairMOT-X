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
import copy
from tqdm import tqdm

from sklearn import metrics
from scipy import interpolate
import numpy as np
from torchvision.transforms import transforms as T
from models.model import create_model, load_model
from datasets.dataset.jde import DetDataset, collate_fn
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from opts import opts
from models.networks.yolox.utils.boxes import postprocess


def test_det(
        opt,
        batch_size=12,
        img_size=(1024, 576),
        iou_thres=0.5,
        print_interval=50,
):
    data_cfg = opt.data_cfg
    with open(data_cfg) as f:
        data_cfg_dict = json.load(f)
    nC = 8
    test_path = data_cfg_dict['test']
    dataset_root = data_cfg_dict['root']
    
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt=opt)
    assert opt.load_model is not None, "No Model to Load for tracking!"
    model = load_model(model, opt.load_model)
    
    try:
        print("Detection Loss Weight: ", model.head.s_det)
        print("ID Loss Weight: ", model.head.s_id)
    except:
        print("Model does not use uncertainty loss.")

    # ----- Set Model to Device & Evaluation Mode
    
    device = opt.device
    model.to(device).eval()

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = DetDataset(dataset_root, test_path, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=8, drop_last=False, collate_fn=collate_fn)
    
    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    
    # print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    
    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
        t = time.time()
        origin_shape = shapes[0]
        width = origin_shape[1]
        height = origin_shape[0]
        
        imgs = imgs.float().to(device=opt.device, non_blocking=True)

        pred, _ = model.forward(imgs)

        # ---- Applies NMS and Returns bboxes
        pred = postprocess(pred, opt.num_classes,
                            conf_thre=opt.det_thre,
                            nms_thre=opt.nms_thre,
                            class_agnostic=True)

        # Compute average precision for each sample
        
        targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
        for si, labels in enumerate(targets):
            seen += 1
            dets = pred[si].cpu().numpy() if pred[si] is not None else None

            if dets is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # If no labels add number of detections as incorrect
            correct = []
            if labels.size(0) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0)
                continue
            else:
                # Scale up the model bboxes to original image
                dets[:, [0,2]] *= width / 1024
                dets[:, [1,3]] *= height / 576
                
                target_cls = labels[:, 0]
                pred_cls = dets[:, -1]

                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = xywh2xyxy(labels[:, 2:6])
                target_boxes[:, 0] *= width
                target_boxes[:, 2] *= width
                target_boxes[:, 1] *= height
                target_boxes[:, 3] *= height
                
                # path = paths[si]
                # img0 = cv2.imread(path)
                # img1 = cv2.imread(path)
                # for t in range(len(target_boxes)):
                #     x1 = int(target_boxes[t, 0])
                #     y1 = int(target_boxes[t, 1])
                #     x2 = int(target_boxes[t, 2])
                #     y2 = int(target_boxes[t, 3])
                #     cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 4)
                # cv2.imwrite('gt.jpg', img0)
                # for t in range(len(dets)):
                #     x1 = int(dets[t, 0])
                #     y1 = int(dets[t, 1])
                #     x2 = int(dets[t, 2])
                #     y2 = int(dets[t, 3])
                #     cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 4)
                # assert cv2.imwrite('pred.jpg', img1)
                
                detected = []
                
                # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                for *pred_bbox, obj_conf, cls in dets:
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and cls == labels[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # Compute Average Precision (AP) per class
            AP, AP_class, R, P = ap_per_class(tp=correct,
                                              conf=dets[:, 4],
                                              pred_cls=pred_cls,  # detections[:, 6]
                                              target_cls=target_cls)

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mAPs.append(AP.mean())
            mR.append(R.mean())
            mP.append(P.mean())

            # Means of all images
            mean_mAP = np.sum(mAPs) / (np.sum(AP_accum_count) + 1E-16)
            mean_R = np.sum(mR) / (np.sum(AP_accum_count) + 1E-16)
            mean_P = np.sum(mP) / (np.sum(AP_accum_count) + 1E-16)

        if batch_i % print_interval == 0:
            pass
            # Print image mAP and running mean mAP
            # print(('%11s%11s' + '%11.3g' * 4 + 's') %
            #(seen, dataloader.dataset.nF, mean_P, mean_R, mean_mAP, time.time() - t))
    # Print mAP per class
    # print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))

    print('AP:', AP_accum / (AP_accum_count + 1E-16), flush=True)
    print('AP:', np.mean(AP_accum / (AP_accum_count + 1E-16)), flush=True)

    # Return mAP
    return mean_mAP, mean_R, mean_P

def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    path = self.img_files[index]
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    return img, img.shape[:2]

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    with torch.no_grad():
        map = test_det(opt, batch_size=16)
