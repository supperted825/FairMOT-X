from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy.core._multiarray_umath import ndarray

import _init_paths
import os
import os.path as osp
import shutil
import cv2
import json
import logging
import argparse
import motmetrics as mm
from tqdm import tqdm
import numpy as np
import torch

from collections import defaultdict
from lib.tracker.multitracker import JDETracker, MCJDETracker
from lib.tracker.YoloTracker import YOLOTracker
from lib.tracker.YoloByteTracker import YOLOBYTETracker

from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
from lib.tracking_utils.utils import mkdir_if_missing

import lib.datasets.yolomot as datasets

from lib.opts import opts


class_names = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))
    

def write_bdd_results(filename, results, img_dim=(720, 1280), bbox_dim=(576, 1024)):

    videoName = filename.split("/")[-1].split(".")[0]
    im_h, im_w = img_dim
    bbox_h, bbox_w = bbox_dim
    
    json_results = [{
            "videoName"     : videoName,
            "name"          : videoName + "-" + str(x+1).rjust(7,"0") + ".jpg",
            "frameIndex"    : x,
            "labels"        : []
        } for x in range(len(results[0]))]
    
    # Each Class Has List of Frame Index
    for class_id, frames in results.items():
        # Each Frame is (frameIndex, list(bboxes), list(track IDs), list(scores))
        for (frame, bboxes, track_ids, scores) in frames:
            for tlwh, track_id in zip(bboxes, track_ids):
                l_dict = {}
                l_dict['id'] = str(track_id)
                l_dict['category'] = class_names[int(class_id)]
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                x1 *= im_w / bbox_w
                x2  *= im_w / bbox_w
                y1 *= im_h / bbox_h
                y2  *= im_h / bbox_h
                l_dict['box2d'] = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                json_results[frame-1]["labels"].append(l_dict)
                
    logger.info('save results to {}'.format(filename))

    with open(filename if ".json" in filename else filename + ".json", 'w') as f:
        json.dump(json_results, f)


def format_dets_dict2dets_list(dets_dict, w, h):
    """
    :param dets_dict:
    :param w: input image width
    :param h: input image height
    :return:
    """
    dets_list = []
    for k, v in dets_dict.items():
        for det_obj in v:
            x1, y1, x2, y2, score, cls_id = det_obj
            center_x = (x1 + x2) * 0.5 / float(w)
            center_y = (y1 + y2) * 0.5 / float(h)
            bbox_w = (x2 - x1) / float(w)
            bbox_h = (y2 - y1) / float(h)

            dets_list.append([int(cls_id), score, center_x, center_y, bbox_w, bbox_h])

    return dets_list

def eval_seq(opt,
             data_loader,
             write_result,
             result_f_name,
             save_dir=None,
             show_image=True,
             frame_rate=30,
             mode='track'):
    """
    :param opt:
    :param data_loader:
    :param data_type:
    :param result_f_name:
    :param save_dir:
    :param show_image:
    :param frame_rate:
    :param mode: track or detect
    :return:
    """
    if save_dir:
        mkdir_if_missing(save_dir)

    # tracker = JDETracker(opt, frame_rate)
    # tracker = YOLOBYTETracker(opt)
    tracker = YOLOTracker(opt)

    timer = Timer()

    results_dict = defaultdict(list)

    frame_id = 0  # frame index
    for path, img, img0 in data_loader:
        # if frame_id % 30 == 0 and frame_id != 0:
            # logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1.0 / max(1e-5, timer.average_time)))

        # --- run tracking
        blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)

        if mode == 'track':  # process tracking
            # ----- track updates of each frame
            timer.tic()

            online_targets_dict = tracker.update_tracking(blob, img0)

            timer.toc()
            # -----

            # collect current frame's result
            online_tlwhs_dict = defaultdict(list)
            online_ids_dict = defaultdict(list)
            online_scores_dict = defaultdict(list)
            for cls_id in range(opt.num_classes):  # process each class id
                online_targets = online_targets_dict[cls_id]
                for track in online_targets:
                    tlwh = track.tlwh
                    t_id = track.track_id
                    score = track.score
                    if tlwh[2] * tlwh[3] > opt.min_box_area:  # and not vertical:
                        online_tlwhs_dict[cls_id].append(tlwh)
                        online_ids_dict[cls_id].append(t_id)
                        online_scores_dict[cls_id].append(score)

            # collect result
            for cls_id in range(opt.num_classes):
                results_dict[cls_id].append((frame_id + 1,
                                             online_tlwhs_dict[cls_id],
                                             online_ids_dict[cls_id],
                                             online_scores_dict[cls_id]))

            # draw track/detection
            if show_image or save_dir is not None:
                if frame_id > 0:
                    online_im: ndarray = vis.plot_tracks(image=img0,
                                                         bbox_dim=(576, 1024),
                                                         tlwhs_dict=online_tlwhs_dict,
                                                         obj_ids_dict=online_ids_dict,
                                                         num_classes=opt.num_classes,
                                                         frame_id=frame_id,
                                                         fps=1.0 / timer.average_time)

        if frame_id > 0:
            if show_image:
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        # update frame id
        frame_id += 1

    # write track/detection results
    if write_result:
        write_bdd_results(result_f_name, results_dict)

    return frame_id, timer.average_time, timer.calls


def main(opt,
         data_root='/data/MOT16/train',
         det_root=None, seqs=('MOT16-05',),
         exp_name='demo',
         save_images=False,
         save_videos=False,
         show_image=True):

    logger.setLevel(logging.INFO)
    
    epochnum = int(opt.load_model.split("_")[-1][:-4])

    exp_root = f"/home/svu/e0425991/FairMOT-X/results/val/{exp_name}"
    result_root = f"/home/svu/e0425991/FairMOT-X/results/val/{exp_name}/epoch{epochnum}"
    mkdir_if_missing(exp_root)
    mkdir_if_missing(result_root)
    
    data_type = 'bdd'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    
    for seq in tqdm(seqs):
        output_dir = osp.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        
        dataloader = datasets.LoadImages(osp.join(data_root, seq))

        result_filename = osp.join(result_root, '{}.json'.format(seq))

        frame_rate = 30
        
        nf, ta, tc = eval_seq(opt, dataloader, True, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

    # ----- timing
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

def FindFreeGPU():
    """
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
    memory_left_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]

    most_free_gpu_idx = np.argmax(memory_left_gpu)
    # print(str(most_free_gpu_idx))
    return int(most_free_gpu_idx)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    opt.device = FindFreeGPU()
    
    val_data = "/hpctmp/e0425991/datasets/bdd100k/bdd100k/images/track/val/"
    seqs = os.listdir(val_data)

    main(opt,
         data_root=val_data,
         seqs=seqs,
         exp_name=opt.exp_id,
         show_image=False,
         save_images=False,
         save_videos=False)
