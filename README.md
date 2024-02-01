# PANPP-mmocr

## News

## Introduction
This project is a re-implementation of the paper [PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text](https://arxiv.org/pdf/2105.00405) using [MMOCR](
https://github.com/open-mmlab/mmocr). PAN++ is an End-to-end framework for arbitrary-shaped text spotting.
The official implementation of the paper can be found [here](https://github.com/whai362/pan_pp.pytorch).

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
    
    git clone https://github.com/ChuanyangGong/PANPP-mmocr.git   # clone from github
    cd PANPP-mmocr
    ```
## Dataset Preparation
1. You can prepare the datasets using mmocr dataset preparation tools. Please refer to [dataset prepare](https://mmocr.readthedocs.io/en/latest/user_guides/dataset_prepare.html#downloading-datasets-and-converting-format) and [dataset_preparer](https://mmocr.readthedocs.io/en/latest/user_guides/data_prepare/dataset_preparer.html) for more details.
    ```commandline
    cd ../../
    # ctw1500
    python tools/dataset_converters/prepare_dataset.py ctw1500 --task textspotting --overwrite-cfg
    
    # icdar2015
    python tools/dataset_converters/prepare_dataset.py icdar2015 --task textspotting --overwrite-cfg
    
    # synthtext
    python tools/dataset_converters/prepare_dataset.py synthtext --task textspotting --overwrite-cfg
    
    # totaltext
    python tools/dataset_converters/prepare_dataset.py totaltext --task textspotting --overwrite-cfg
    
    # cocotextv2
    python tools/dataset_converters/prepare_dataset.py cocotextv2 --task textspotting --overwrite-cfg
    ```
    
    **Note:** Ensure that the working directory is `mmocr` when executing the above commands. MMOCR also support other datasets, please refer to [datasetzoo](https://mmocr.readthedocs.io/en/latest/user_guides/data_prepare/datasetzoo.html) for more details.
    
    Then, link the dataset to the `PANPP-mmocr`.
   ```commandline
   cd projects/PANPP-mmocr
   ln -s ../../data ./data
   ```

    Finally, your directory structure should look like this:
    ```
    PANPP-mmocr
    ├── data
    │   ├── ctw1500
    │   │   ├── lexicons
    │   │   ├── textdet_imgs
    │   │   ├── textspotting_test.json
    │   │   └── textspotting_train.json
    │   ├── icdar2015
    │   │   ├── lexicons
    │   │   ├── textdet_imgs
    │   │   ├── textspotting_test.json
    │   │   └── textspotting_train.json
    │   ├── synthtext
    │   │   ├── textdet_imgs
    │   │   └── textspotting_train.json
    │   ├── totaltext
    │   │   ├── lexicons
    │   │   ├── textdet_imgs
    │   │   ├── textspotting_test.json
    │   │   └── textspotting_train.json
    │   └── cocotextv2
    │       ├── textdet_imgs
    │       ├── textspotting_test.json
    │       └── textspotting_train.json
    ├── ...
    ```

2. Custom Dataset Preparation
   - To prepare your custom dataset, follow the guidelines provided in this [doc](https://mmocr.readthedocs.io/en/latest/basic_concepts/datasets.html#dataset-classes-and-annotation-formats). The annotation format for MMOCR's dataset is clearly explained within the document.

3. Customizing Character Sets
   - If your custom dataset involves non-English characters or you wish to modify the character set, it's essential to provide your own character set file. Refer to `dicts/english_characters.txt` for an example. Ensure that your custom character set file accurately represents the characters present in your dataset.

## Training
1. To train on multiple GPUs, e.g. 2 GPUs, run the following command:
   ```commandline
   ../../tools/dist_train.sh config/panpp/panpp_resnet18_fpem-ffm_300k_en-joint-train.py 2 --auto-scale-lr
   ```
2. To resume training from a checkpoint, use the following command:
   ```commandline
   ../../tools/dist_train.sh config/panpp/panpp_resnet18_fpem-ffm_300k_en-joint-train.py 2 --auto-scale-lr --resume
   ```
3. If you intend to pretrain a model and fine-tune it afterward, load the pre-trained model in the fine-tuning configuration. Add the `load_from` parameter in the config file to achieve this. For example, in `config/panpp/panpp_resnet18_fpem-ffm_14k_ic15.py`:
   ```python
   load_from = 'your_pretrained_model.pth'
   ```
