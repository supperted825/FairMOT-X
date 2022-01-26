# FairMOT-X

FairMOT-X is a multi-class multi object tracker based on [FairMOT](https://github.com/ifzhang/FairMOT), which has been tailored for training on the BDD100K MOT Dataset. It makes use of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) as the detector from end-to-end, and uses deformable convolutions (DCN) to perform feature fusion of PAFPN outputs for reID embedding learning.

<br>

<p align="center">
    <img src="./media/FairMOT-X.png" width="100%", height="100%"<br/>
    <em> </br> Overview of FairMOT-X Structure</em>
</p>

## Tracking Performance

### Results on BDD100K Dataset

Results on other variants will be added soon.

| Variant | FPS | mMOTA | mMOTP | mIDF1 |
| - | - | - | - | - |
| YOLOX-S | 36.1 | 16.7 | 67.1 | 25.6 |
| YOLOX-M | 32.7 | 18.4 | 68.0 | 27.5 |

### Video Demos from BDD100K MOT

<br>

<p align="center">
    <img src="./media/bdd1.gif" width="80%", height="80%"<br/>
</p>

<p align="center">
    <img src="./media/bdd2.gif" width="80%", height="80%"<br/>
</p>



## Installation

Please refer to the [FairMOT installation instructions](https://github.com/ifzhang/FairMOT#Installation) to install the required dependencies.

## Train & Demo

The following command runs training with YOLOX-M as the detector. The corresponding network depth and width must be specified, or the default (YOLOX-L) will be used.

```
python3 ./src/train.py mot \
    --exp_id yolo-m --yolo_depth 0.67 --yolo_width 0.75 \
    --lr 7e-4 --lr_step 2 \
    --reid_dim 128 --augment --mosaic \
    --batch_size 16 --gpus 0 
```

To run the demo after training, you can refer to the following example:

```
python3 -W ignore ./src/demo.py mot \
	--load_model path_to_model.pth \
    --input_video path_to_video_or_folder_of_images \
    --reid_dim 128 --yolo_depth 0.67 --yolo_width 0.75
```

## Acknowledgement

This project heavily uses code from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), the original [FairMOT](https://github.com/ifzhang/FairMOT), as well as [MCMOT](https://github.com/CaptainEven/MCMOT) and [YOLOv4 MCMOT](https://github.com/CaptainEven/YOLOV4_MCMOT).
