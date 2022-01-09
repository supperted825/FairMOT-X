import glob
from io import DEFAULT_BUFFER_SIZE
import math
import os
import json
import random
import shutil
import copy
import time
from collections import defaultdict
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from lib.utils.utils import xyxy2xywh, xywh2xyxy, ltwh2xywh

cls2id = {
    'pedestrian': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7
}

id2cls = {
    0: 'pedestrian',
    1: 'rider',
    2: 'car',
    3: 'truck',
    4: 'bus',
    5: 'train',
    6: 'motorcycle',
    7: 'bicycle'
}

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


class LoadImages:
    def __init__(self, path, img_size=(1024, 576)):
        """
        :param path:
        :param img_size:
        """
        self.frame_rate = 10  # no actual meaning here

        if type(path) == str:
            if os.path.isdir(path):
                image_format = ['.jpg', '.jpeg', '.png', '.tif']
                self.files = sorted(glob.glob('%s/*.*' % path))
                self.files = list(filter(lambda x: os.path.splitext(x)[
                                                       1].lower() in image_format, self.files))
            elif os.path.isfile(path):
                self.files = [path]
        elif type(path) == list:
            self.files = path

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        if self.count == self.nF:
            raise StopIteration

        img_path = self.files[self.count]

        # Read image
        img_0 = cv2.imread(img_path)  # BGR
        assert img_0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _ = letterbox(img_0, (self.height, self.width))

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img_0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img_0 = cv2.imread(img_path)  # BGR
        assert img_0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # Normalize RGB: BGR -> RGB and H×W×C -> C×H×W
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img_0

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe=0, img_size=416):
        self.img_size = img_size

        if pipe == '0':
            pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'  # IP traffic camera
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=416):
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(0 if s == '0' else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


class YOLOMOT(Dataset):  # for training/testing
    def __init__(self,
                 path,
                 img_size=(576, 1024),
                 num_classes=8,
                 single_cls=False,
                 opt=None):
        """
        :param path:
        :param img_size:
        :param batch_size:
        :param augment:
        :param single_cls:
        """
        
        # Path is .train or .val file
        assert os.path.isfile(path), 'File not found %s. See %s' % (path, help_url)
        
        # ------ Check for QuickLoad of Labels & IDs
        labels    = "/hpctmp/e0425991/datasets/bdd100k/bdd100k/MOT/cache/cached_labels.npy"
        tid_num   = "/hpctmp/e0425991/datasets/bdd100k/bdd100k/MOT/tid_num.json"

        labelsval = "/hpctmp/e0425991/datasets/bdd100k/bdd100k/MOT/cache/cached_labelsval.npy"
        tidnumval = "/hpctmp/e0425991/datasets/bdd100k/bdd100k/MOT/tid_numval.json"
        
        # Get List of Img Files
        with open(path, 'r') as f:
            self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
                            if os.path.splitext(x)[-1].lower() in img_formats]
        
        # Get List of Label Files
        self.label_files = [x.replace('images', 'labels_with_ids').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        # ----- Calculate Dataset Parameters
        n = len(self.img_files)

        self.n = n
        self.default_input_wh = img_size
        self.num_classes = num_classes
        self.augment = opt.augment
        self.mosaic = opt.mosaic
        self.max_objs = opt.K
        self.img_size = img_size

        # ----- Cache labels
        self.labels = [np.zeros((0, 6), dtype=np.float32)] * n

        # ------ Check for QuickLoad of Image Labels
        if os.path.exists(labels) and os.path.exists(tid_num) and not opt.val:
            print("Loading cached labels & ID Dict...")
            self.labels = np.load(labels, allow_pickle=True)
            with open(tid_num) as json_file:
                self.tid_num = json.load(json_file)
        
        elif os.path.exists(labelsval) and os.path.exists(tidnumval) and opt.val:
            print("Loading cached validation labels & ID Dict...")
            self.labels = np.load(labelsval, allow_picke=True)
            with open(tidnumval) as json_file:
                self.tid_num = json.load(json_file)
        
        else:
            # ----- Generate ID Counts
            print("Caching Labels & Generating ID Counts...")
            
            self.tid_num = OrderedDict()
            self.tid_start_index = OrderedDict()
            
            for ds, label_paths in self.label_files.items():  # 每个子数据集
                max_ids_dict = defaultdict(int)  # cls_id => max track id

                # 子数据集中每个label
                for lp in tqdm(label_paths):
                    if not os.path.isfile(lp):
                        print('[WARNING] : Invalid Label File {}.'.format(lp))
                        continue

                    lb = np.loadtxt(lp)
                    if len(lb) < 1:  # 空标签文件
                        continue

                    lb = lb.reshape(-1, 6)

                    assert (lb >= 0).all(), 'Negative Labels: %s' % file
                    assert (lb[:, 2:] <= 1).all(), 'Non-normalized or Out of Bounds Coordinates: %s' % file

                    if single_cls:
                        lb[:, 0] = 0  # force dataset into single-class mode: turn mc to sc

                    self.labels[i] = lb
                
                    for item in lb:  # label中每一个item(检测目标)
                        if item[1] > max_ids_dict[int(item[0])]:  # item[0]: cls_id, item[1]: track id
                            max_ids_dict[int(item[0])] = item[1]

                # track id number
                self.tid_num[ds] = max_ids_dict  # 每个子数据集按照需要reid的cls_id组织成dict
                
            # ----- save dicts to json to save time in future
            if opt.val:
                print("Writing cached labels & max ID dict to JSON...")
                np.save(labels, self.labels)
                with open(tidnumval, 'w', encoding='utf-8') as f:
                    json.dump(self.tid_num, f, ensure_ascii=False, indent=4)    
            else:
                print("Writing cached validation labels & max ID dict to JSON...")
                np.save(labelsval, self.labels)
                with open(tid_num, 'w', encoding='utf-8') as f:
                    json.dump(self.tid_num, f, ensure_ascii=False, indent=4)  
                    
        # @even: for MCMOT training
        self.tid_start_idx_of_cls_ids = defaultdict(dict)
        last_idx_dict = defaultdict(int)  # 从0开始
        for k, v in self.tid_num.items():  # 统计每一个子数据集
            for cls_id, id_num in v.items():  # 统计这个子数据集的每一个类别, v是一个max_ids_dict
                self.tid_start_idx_of_cls_ids[k][int(cls_id)] = last_idx_dict[int(cls_id)]
                last_idx_dict[int(cls_id)] += id_num

        # Generate nID Dict for Building reID Classifier
        self.nID_dict = defaultdict(int)
        for k, v in last_idx_dict.items():
            self.nID_dict[int(k)] = int(v)  # 每个类别的tack ids数量

        for k, v in sorted(self.nID_dict.items()):
            class_name = id2cls[k]
            print('* Total {:d} IDs of {}'.format(v, class_name))

        # print('start index', self.tid_start_index)
        for k, v in sorted(self.tid_start_idx_of_cls_ids.items()):
            for cls_id, start_idx in sorted(v.items()):
                print('* Start index of dataset {} class {} is {}'
                        .format(k, cls_id, start_idx))


    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, idx):
        
        if self.mosaic:
            img, labels, track_ids = load_mosaic_with_ids(self, idx)
        
        else:
            # ----- Load Image, Labels & Track IDs
            # Label Format: Class, Track ID, Normalised ltwh
            img, (h, w) = load_image(self, idx)
            resize_ratio = self.img_size[0] / img.shape[0]
            if img.shape[:2] != self.img_size:
                img = cv2.resize(img, self.img_size[::-1])
            
            labels = []
            x = self.labels[idx][:, [0, 2, 3, 4, 5]]  # Skip Loading Track IDs
            if x.size > 0:
                # Normaliseded xywh to pixel xyxy format:
                # For compatibility with random_affine function
                labels = x.copy()
                labels[:, 1] = resize_ratio * w * (x[:, 1] - x[:, 3] / 2)      # x1 = ct - w / 2
                labels[:, 2] = resize_ratio * h * (x[:, 2] - x[:, 4] / 2)      # y1 = ct + h / 2
                labels[:, 3] = resize_ratio * w * (x[:, 1] + x[:, 3] / 2)      # x2 = ct + w / 2
                labels[:, 4] = resize_ratio * h * (x[:, 2] + x[:, 4] / 2)      # y2 = ct - h / 2
            
            # Now We Load Track IDs
            track_ids = self.labels[idx][:, 1]
        
        if self.augment:
            if not self.mosaic:
                # Random Affine & Augment ColourSpace
                img, labels, track_ids = random_affine_with_ids(img, labels, track_ids)
            augment_hsv(img)

        # Number of Labels
        nL = len(labels)
        
        # Track ID starts from 1 (not 0)
        track_ids -= 1  

        # ----- Further Augmentations
        if nL:

            # Convert back from xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            
            # Normalise Again for Easy Flipping
            labels[:, [2,4]] /= self.img_size[0]
            labels[:, [1,3]] /= self.img_size[1]

        if self.augment:
            
            # Random Horizontal Flipping
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # Random Vertical Flipping
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
        
        # ----- Final BBOX Ground Truths
        
        # Initialise Detection BBox Outs
        det_labels_out = np.zeros((nL, 6))  # Additional column0 means item_i in the batch
        track_ids_out  = np.zeros((nL, 2))

        if nL:
            # Scale Normalised to Pixel XYWH for Detection
            det_labels_out[:, 1] = labels[:, 0]
            det_labels_out[:, [2, 4]] = labels[:, [1, 3]] * self.img_size[1]
            det_labels_out[:, [3, 5]] = labels[:, [2, 4]] * self.img_size[0]
            
            # Track IDs to be Returned
            track_ids_out[:, 1] = torch.from_numpy(track_ids).long()

        # Write Test Images with bbox to File
        # test_image(img, det_labels_out[:, 2:])
        # exit(-1)

        # ------ Image Transformations - BGR to RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), torch.from_numpy(det_labels_out), torch.from_numpy(track_ids_out)
    
    @staticmethod
    def collate_fn(batch):
        img, label, track_ids = zip(*batch)
        # Add Batch Index as First Element of Labels & Track IDs
        for i, (l, tid) in enumerate(zip(label, track_ids)):
            l[:, 0] = i
            tid[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), torch.cat(track_ids, 0)
    
    def shuffle(self):
        # Shuffle the dataset
        np.random.shuffle(self.img_files)
        
        # Reindex Corresponding Label Files
        self.label_files = [x.replace('images', 'labels_with_ids')
                                    .replace('.png', '.txt')
                                    .replace('.jpg', '.txt')
                                for x in self.img_files]


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    path = self.img_files[index]
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    return img, img.shape[:2]


def test_image(img, labels, path="/home/svu/e0425991/FairMOT-X", name="test.jpg"):
    img = np.ascontiguousarray(img).copy()
    for (x, y, w, h) in labels:
        x1 = int(x-w/2)
        y1 = int(y-h/2)
        x2 = int(x+w/2)
        y2 = int(y+h/2)
        cv2.rectangle(img, pt1=(x1,y1), pt2=(x2,y2), color=(0, 0, 255), thickness=1)
    assert cv2.imwrite(os.path.join(path, name), img)
    

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def pad_resize_ratio(img, net_w, net_h):
    """
    :param img:
    :param net_w:
    :param net_h:
    :return:
    """
    img = np.array(img)  # H x W x channels
    H, W, channels = img.shape

    if net_h / net_w < H / W:  # padding w
        new_h = int(net_h)
        new_w = int(net_h / H * W)

        pad = (net_w - new_w) // 2

        left = round(pad - 0.1)
        right = round(pad + 0.1)

        top, bottom = 0, 0
    else:  # padding w
        new_h = int(net_w / W * H)
        new_w = int(net_w)

        pad = (net_h - new_h) // 2

        left, right = 0, 0

        top = round(pad - 0.1)
        bottom = round(pad + 0.1)

    img_resize = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)

    # add border
    img_out = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=127)
    return img_out


def pad_resize_img_square(img, square_size):
    """
    :param img: RGB image
    :return: square image
    """
    img = np.array(img)  # H x W x channels
    H, W, channels = img.shape
    dim_diff = np.abs(H - W)

    # upper(left) and lower(right) padding
    pad_lu = dim_diff // 2  # integer division
    pad_rd = dim_diff - pad_lu

    # determine padding for each axis: H, W, channels
    pad = ((pad_lu, pad_rd), (0, 0), (0, 0)) if H <= W else \
        ((0, 0), (pad_lu, pad_rd), (0, 0))

    # do padding(0.5) and normalize
    img = np.pad(img,
                 pad,
                 'constant',
                 constant_values=127.5)  # / 255.0
    img = cv2.resize(img,
                     (square_size, square_size),
                     cv2.INTER_LINEAR)
    # img.tofile('/mnt/diskb/even/img.bin')
    return img


def letterbox(img,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scaleFill=False,
              scaleup=True):
    """
    :param img:
    :param new_shape:
    :param color:
    :param auto:
    :param scaleFill:
    :param scaleup:
    :return:
    """
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)


def random_affine_with_ids(img,
                           targets,
                           track_ids,
                           degrees=2.5,
                           translate=0.1,
                           scale=0.1,
                           shear=2.5,
                           border=0):
    """
    :param img:
    :param targets:
    :param track_ids:
    :param degrees:
    :param translate:
    :param scale:
    :param shear:
    :param border:
    :return:
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:  # targets = [cls, xyxy]
        targets = []
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # ----- Transform Label Coordinates
    n = len(targets)
    if n:
        # Warp Points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # Create New Boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # Reject Warped Points Outside Image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        track_ids = track_ids[i]
        targets[:, 1:5] = xy[i]

    return img, targets, track_ids


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:  # targets = [cls, xyxy]
        targets = []
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def load_mosaic_with_ids(self, index):
    """
    :param self:
    :param index:
    :param track_ids:
    :return:
    """

    labels4, track_ids_4 = [], []
    net_in_h, net_in_w = self.img_size
    
    # Randomly Generate Mosaic Center (Coordinates in Final Image)
    xc = int((random.uniform(0.35, 0.65) * net_in_w))
    yc = int((random.uniform(0.35, 0.65) * net_in_h))
    
    # Grab 3 Other Random Indices
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]
    
    for i, index in enumerate(indices):
        
        # Load Image
        img, (h, w) = load_image(self, index)
        labels = self.labels[index][:, [0,2,3,4,5]]
        track_ids = self.labels[index][:, 1]

        # Generate Coordinates for Each Corner
        # On First Run, Initialize Base Array with Arbitrary Pixel Value of 114
        
        if i == 0: # top left
            img4 = np.full((net_in_h, net_in_w, 3), 114, dtype=np.uint8)
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        elif i == 1:  # top right
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, net_in_w), yc
        elif i == 2:  # bottom left
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(yc + h, net_in_h)
        elif i == 3:  # bottom right
            x1, y1, x2, y2 = xc, yc, min(xc + w, net_in_w), min(yc + h, net_in_h)

        # Fitted Image Dimensions, Resize & Place
        new_h = y2 - y1
        new_w = x2 - x1
        img = cv2.resize(img, (new_w, new_h))
        img4[y1:y2, x1:x2] = img
        
        # Take the Opportunity to Convert Normalised XYWH to Pixel XYXY
        for ltwh_bbox in labels:
            bbox = np.zeros(5)
            bbox[0] = ltwh_bbox[0]
            bbox[1] = x1 + (ltwh_bbox[1] - ltwh_bbox[3] / 2) * new_w
            bbox[2] = y1 + (ltwh_bbox[2] - ltwh_bbox[4] / 2) * new_h
            bbox[3] = bbox[1] + ltwh_bbox[3] * new_w
            bbox[4] = bbox[2] + ltwh_bbox[4] * new_h
            labels4.append(bbox)

        track_ids_4.extend(track_ids)

    # Track IDs Start from 0, not 1
    labels4 = np.array(labels4)
    track_ids_4 = np.array(track_ids_4)

    # Standard Augmentation
    img4, labels4, track_ids = random_affine_with_ids(img4, labels4, track_ids_4)

    return img4, labels4, track_ids


def cutout(image, labels):
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def reduce_img_size(path='../data/sm4/images',
                    img_size=1024):  # from evaluate_utils.datasets import *; reduce_img_size()
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + '_reduced'  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob('%s/*.*' % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            fnew = f.replace(path, path_new)  # .replace(Path(f).suffix, '.jpg')
            cv2.imwrite(fnew, img)
        except:
            print('WARNING: image failure %s' % f)


def convert_images2bmp():  # from evaluate_utils.datasets import *; convert_images2bmp()
    # Save images
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    # for path in ['../coco/images/val2014', '../coco/images/train2014']:
    for path in ['../data/sm4/images', '../data/sm4/background']:
        create_folder(path + 'bmp')
        for ext in formats:  # ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            for f in tqdm(glob.glob('%s/*%s' % (path, ext)), desc='Converting %s' % ext):
                cv2.imwrite(f.replace(ext.lower(), '.bmp').replace(path, path + 'bmp'), cv2.imread(f))

    # Save labels
    # for path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
    for file in ['../data/sm4/out_train.txt', '../data/sm4/out_test.txt']:
        with open(file, 'r') as f:
            lines = f.read()
            # lines = f.read().replace('2014/', '2014bmp/')  # coco
            lines = lines.replace('/images', '/imagesbmp')
            lines = lines.replace('/background', '/backgroundbmp')
        for ext in formats:
            lines = lines.replace(ext, '.bmp')
        with open(file.replace('.txt', 'bmp.txt'), 'w') as f:
            f.write(lines)


def recursive_dataset2bmp(dataset='../data/sm4_bmp'):  # from evaluate_utils.datasets import *; recursive_dataset2bmp()
    # Converts dataset to bmp (for faster training)
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = a + '/' + file
            s = Path(file).suffix
            if s == '.txt':  # replace text
                with open(p, 'r') as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, '.bmp')
                with open(p, 'w') as f:
                    f.write(lines)
            elif s in formats:  # replace image
                cv2.imwrite(p.replace(s, '.bmp'), cv2.imread(p))
                if s != '.bmp':
                    os.system("rm '%s'" % p)


def imagelist2folder(path='data/coco_64img.txt'):  # from evaluate_utils.datasets import *; imagelist2folder()
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            os.system('cp "%s" %s' % (line, path[:-4]))
            print(line)


def create_folder(path='./new_folder'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
