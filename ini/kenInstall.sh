#!/bin/bash
#
git clone https://github.com/sniklaus/3d-ken-burns.git
wget -O ./3d-ken-burns/models/disparity-estimation.pytorch http://content.sniklaus.com/kenburns/network-disparity.pytorch
wget -O ./3d-ken-burns/models/disparity-refinement.pytorch http://content.sniklaus.com/kenburns/network-refinement.pytorch
wget -O ./3d-ken-burns/models/pointcloud-inpainting.pytorch http://content.sniklaus.com/kenburns/network-inpainting.pytorch
pip install chainer
pip install gevent
pip3 install imageio==2.4.1
cd /content/3d-ken-burns
rm images/README.md
mkdir -p results
