from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import logging
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch

my_visible_devs = '0'  # '0, 3'  # 设置可运行GPU编号
os.environ['CUDA_VISIBLE_DEVICES'] = my_visible_devs
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

import cv2
import shutil
import numpy as np
import os.path as osp
import torch.nn.functional as F
from collections import defaultdict
from lib.opts import opts  # import opts
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.log import logger
import lib.datasets.dataset.bdd100k as datasets
from track import eval_seq, eval_imgs_output_dets
from lib.datasets.dataset.jde import letterbox
from lib.models.model import create_model, load_model
from lib.models.decode import mot_decode
from lib.models.utils import _tranpose_and_gather_feat
from lib.tracker.multitracker import map2orig
from lib.tracking_utils.visualization import plot_detects
from lib.utils.utils import select_device

logger.setLevel(logging.INFO)


# find the GPU idx with the largest remaining memory
def FindFreeGPU():
    """
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
    memory_left_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]

    most_free_gpu_idx = np.argmax(memory_left_gpu)

    return int(most_free_gpu_idx)


def run_demo(opt):
    """
    :param opt:
    :return:
    """
    result_root = "/home/svu/e0425991/MCMOT/results"
    mkdir_if_missing(result_root)

    # clear existing frame results
    frame_res_dir = result_root + '/frame'
    if os.path.isdir(frame_res_dir):
        shutil.rmtree(frame_res_dir)
        os.makedirs(frame_res_dir)
    else:
        os.makedirs(frame_res_dir)
        
    if os.path.isfile(opt.input_video):
        videoname = opt.input_video.split("/")[-1].split(".")[0]
        dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    else:
        videoname = opt.input_video.split("/")[-1]
        if videoname == "":
            videoname = opt.input_video.split("/")[-2]

        dataloader = datasets.LoadImages(opt.input_video, opt.img_size)

    result_file_name = os.path.join(result_root, '{}'.format(videoname))
    
    try:
        frame_rate = dataloader.frame_rate
    except:
        frame_rate = 30
        
    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')

    opt.device = str(FindFreeGPU())
    print('Using GPU: {:s}'.format(opt.device))
    device = select_device(device='cpu' if not torch.cuda.is_available() else opt.device)
    opt.device = device
    
    eval_seq(opt=opt,
            data_loader=dataloader,
            data_type='bdd',
            result_f_name=result_file_name,
            save_dir=frame_dir,
            show_image=False,
            frame_rate=frame_rate,
            mode='track')

    except Exception as e:
        logger.info(e)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, '{}_result.mp4'.format(videoname))
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000M -c:v mpeg4 {}' \
            .format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)
        os.system("rm -r /home/svu/e0425991/FairMOT-BDD100K/results/frame")

if __name__ == '__main__':
    opt = opts().init()
    run_demo(opt)
