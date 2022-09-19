# Learning Monocular Depth in Dynamic Scenes via Instance-Aware Projection Consistency

## Introduction
This projective is to prepare training data for video data given camera intrinsics. Please check [the original repository](https://github.com/SeokjuLee/Insta-DM) to ​know the details.

## Environment Setup

Setup a conda environment
```
conda create -n insta_dm python=3.7.4
conda activate insta_dm
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio=0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pytorch-sparse -c pyg
pip install -r requirements.txt # revise matplotlib==3.4.3
```

Install the dependencies for optical flow generation. Please ensure the version of your local CUDA toolkit newer or equal to 10.2.
```
cd ./pwc_net/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user
```

## Data Preparation from Videos

```
# generate images and instance segmentation masks from videos
python gen_data.py --dataset_name video --dataset_dir $FOLDER_OF_VIDEOS \
                   --save_dir $FOLDER_TO_SAVE_DATA --save_img_ext jpg \
                   --intrinsics intrinsics.txt \ # flattened 9 entries of the camera matrix 
                   --img_height $HEIGHT --img_width $WIDTH --mask instance \
                   --single_process --del_static_frames 

# generate optical flows
python gen_optical_flows.py --root_dir $FOLDER_TO_SAVE_DATA \
                            --img_ext jpg \
```
The content of an example intrinsics.txt is shown below. It contains the 9 entries of a flattented camera matrix.
```
1344.8 0.0 640.0 0.0 1344.8 360.0 0.0 0.0 1.0
```

## Training

Please download the checkpoints pretrained on KITTI or Cityscapes from [the official site](https://github.com/SeokjuLee/Insta-DM#models) to $PRETRAINED

```
CUDA_VISIBLE_DEVICES=0 python train.py $FOLDER_TO_SAVE_DATA \
        --pretrained-disp $PRETRAINED/resnet18_disp_kt.tar \
        --pretrained-ego-pose $PRETRAINED/resnet18_ego_kt.tar \
        --pretrained-obj-pose $PRETRAINED/resnet18_obj_kt.tar \
        --data_file_structure custom \
        -b 4 -p 2.0 -c 1.0 -s 0.1 -o 0.02 -mc 0.1 -hp 0 -dm 0 -mni 5 \
        --epoch-size 1000 --with-ssim --with-mask --with-auto-mask \
        --name $MODEL_NAME  # models would be saved in checkpoints/$MODEL_NAME within the project folder
```

## Inference
```
CUDA_VISIBLE_DEVICES=1 python demo.py \
        --data $DATA_DIR \ # path to a folder containing images, optical flows, and segmentation masks
        --pretrained-disp $PRETRAINED/best/dispnet_model_best.pth.tar \
        --pretrained-ego-pose $PRETRAINED/best/ego_pose_model_best.pth.tar \
        --pretrained-obj-pose $PRETRAINED/best/obj_pose_model_best.pth.tar \
        --data_file_structure 'custom' \
        --mni 1 \
        --name NAME_TO_SAVE \ # name of the folder to save in outputs
        --save-fig \
```
