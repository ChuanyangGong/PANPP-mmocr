# PANPP-mmocr

## Installation
1. Clone the code and change the directory
```commandline
git clone https://gitee.com/g847714121/PANPP-mmocr.git       # clone from gitee
git clone https://github.com/ChuanyangGong/PANPP-mmocr.git   # clone from github
cd PANPP-mmocr
```
2. Create a conda environment and activate it
```commandline
conda create --name mmocr python=3.8 -y
conda activate mmocr
```
3. Install Pytorch
```commandline
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch
```
4. Install MMEngine, MMCV, MMDetection and MMOCR using MIM.
```commandline
pip install -U openmim
mim install mmengine==0.8.5
mim install mmcv==2.0.1
mim install mmdet==3.1.0
mim install mmocr==1.0.1
```