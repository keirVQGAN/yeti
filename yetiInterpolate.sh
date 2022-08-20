#!/bin/bash
pip install gdown --upgrade
pip install tensorflow-datasets==4.4.0 tensorflow-addons==0.15.0 absl-py==0.12.0 gin-config==0.5.0 parameterized==0.8.1 mediapy==1.0.3 scikit-image==0.19.1 apache-beam==2.34.0
git clone https://github.com/google-research/frame-interpolation frame_interpolation
cd frame_interpolation
gdown 1C1YwOo293_yrgSS8tAyFbbVcMeXxzftE
unzip "/content/frame_interpolation/pretrained_models-20220214T214839Z-001.zip"
cd /content/