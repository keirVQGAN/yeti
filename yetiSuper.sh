#!/bin/bash
!git clone https://github.com/xinntao/Real-ESRGAN.git               &> /dev/null
%cd /content/Real-ESRGAN
!pip install basicsr                                              &> /dev/null
!pip install -r requirements.txt                                  &> /dev/null
!python setup.py develop                                                      
!wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models