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

## Acknowledgement

This project heavily uses code from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), the original [FairMOT](https://github.com/ifzhang/FairMOT), as well as [MCMOT](https://github.com/CaptainEven/MCMOT) and [YOLOv4 MCMOT](https://github.com/CaptainEven/YOLOV4_MCMOT).
