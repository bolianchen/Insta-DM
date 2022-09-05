# Learning Monocular Depth in Dynamic Scenes via Instance-Aware Projection Consistency

## Introduction
This projective is to prepare training data for video data given camera intrinsics. Please check [the original repository](https://github.com/SeokjuLee/Insta-DM) to ​know the details.

## Environment Setup
```
conda create -n insta_dm python=3.7.4
conda activate insta_dm
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio=0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pytorch-sparse -c pyg
pip install -r requirements.txt # revise matplotlib==3.4.3
```
