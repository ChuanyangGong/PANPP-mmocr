# PANPP-mmocr

## News

## Installation
1. Create a conda environment and activate it
```commandline
conda create --name mmocr python=3.8 -y
conda activate mmocr
```
2. Install Pytorch
```commandline
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch
```
3. Install MMEngine, MMCV, MMDetection using MIM.
```commandline
pip install -U openmim
mim install mmengine==0.8.5
mim install mmcv==2.0.1
mim install mmdet==3.1.0
```
4. Install mmocr from source
```commandline
git clone https://github.com/open-mmlab/mmocr.git            # clone from github
git clone https://gitee.com/open-mmlab/mmocr.git             # clone from gitee
cd mmocr
```
I strongly recommend using version 1.0.1 of MMOCR to avoid potential issues caused by version discrepancies. Please note that the code cloned from Gitee may not have been updated to version 1.0.1 yet.
```commandline
git checkout tags/v1.0.1 -b v1.0.1
```
Install mmocr
```commandline
pip install -v -e .
```
5. Change the directory to `projects` clone the code of this project, and navigate into the project directory.
```commandline
cd projects

git clone https://gitee.com/g847714121/PANPP-mmocr.git       # clone from gitee
git clone https://github.com/ChuanyangGong/PANPP-mmocr.git   # clone from github

cd PANPP-mmocr
```
## Prepare datasets
