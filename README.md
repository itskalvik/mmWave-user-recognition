# Gait Based User Recognition from mmWave Radar Data
Deep learning methods for user recognition from [mmWave radars](https://www.ti.com/sensors/mmwave-radar/overview.html). We used [IWR1642 single-chip 76-GHz to 81-GHz mmWave sensor integrating DSP and MCU evaluation module](https://www.ti.com/tool/IWR1642BOOST) along with [DCA1000EVM](https://www.ti.com/tool/DCA1000EVM) board to collect gait data(private dataset). The methods in this repo convert
radar data to spectrograms and use deep learning methods for user recognition. The methods include, vanilla deep convolutional neural networks and domain adaption methods. The domain adaption methods were used to reduce the amount of supervised data needed to train a model and generalize to new data collection environments without the need to explicitly train on data from the new environments.

## Directory Structure
- `models/`: Deep learning models/experiments
  - `resnet.py`: Vanilla Resnet base model imported by some models
  - `resnet_amca.py`: Resnet model with Constrictive Annular Loss(metric learning method) imported by some models
  - `utils.py`: Arbitrary methods used by most model scripts (loss functions, data preprocessing, confusion matrices, ...)
- `preprocess/`: Script to generate spectrograms from raw data collected from radar
- `tools/`: Arbitrary tools to plot data, list dataset files, generate
            embeddings from pretrained models
  - `mmwave_studio_scripts/`: Scripts used to collect radar data from [TI mmWave Studio](https://www.ti.com/tool/MMWAVE-STUDIO) along with video/depth data from [Microsoft Kinect Azure](https://azure.microsoft.com/en-us/services/kinect-dk/) camera. Use `*3d` files to record camera data along with radar data.
        
## Citation
Please cite the following paper if you use this repository in your research.
```
@inproceedings{janakaraj2019star,
  title={STAR: Simultaneous Tracking and Recognition through Millimeter Waves and Deep Learning},
  author={Janakaraj, Prabhu and Jakkala, Kalvik and Bhuyan, Arupjyoti and Sun, Zhi and Wang, Pu and Lee, Minwoo},
  booktitle={2019 12th IFIP Wireless and Mobile Networking Conference (WMNC)},
  pages={211--218},
  year={2019},
  organization={IEEE}
```
