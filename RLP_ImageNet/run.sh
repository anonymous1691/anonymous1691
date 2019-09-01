#!/usr/bin/env sh
set -e

cd /mnt/cephfs_new_wj/cv/xiaxin/fbnet_v1/flops-counter.pytorch
sudo python3 setup.py install
cd /mnt/cephfs_new_wj/cv/xiaxin/GAL_ImageNet/GAL_KD3
##pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali
pip3 install nvidia_dali-0.12.0-819488-cp36-cp36m-manylinux1_x86_64.whl
##pip3 install torch==1.1.0
#pip3 install tensorboardX
#python3 main_res_kse.py --teacher_dir resnet50-19c8e357.pth
##python3 supernet_main_file.py --train_or_sample sample --architecture_name fbnet_imagenet1000 --hardsampling_bool_value True
##python3 supernet_main_file.py --train_or_sample train
#cd /mnt/cephfs_hl/cv/xiaxin/GAL_ImageNet/GAL_KD3
#pip3 install nvidia_dali-0.12.0-819488-cp36-cp36m-manylinux1_x86_64.whl


cd /mnt/cephfs_new_wj/cv/xiaxin/GAL_ImageNet/github/RLP_ImageNet
#CUDA_VISIBLE_DEVICES=0 python3 finetune_resnet.py --refine /mnt/cephfs_new_wj/cv/xiaxin/GAL_ImageNet/GAL_KD3/experiment_cp_sign0.4/checkpoint/model_59.pt

CUDA_VISIBLE_DEVICES=0 python3 finetune_resnet.py --refine /mnt/cephfs_new_wj/cv/xiaxin/GAL_ImageNet/GAL_KD3/experiment_cp_sign/checkpoint/model_45.pt

#python3 finetune_resnet.py --refine ../GAL_KD3/26_11_prune2000/checkpoint/model_30.pt


##
##cp -r /mnt/cephfs_new_wj/cv/xiaxin/GAL_ImageNet/data/CIFAR /dev/shm/
#cd /mnt/cephfs_new_wj/cv/xiaxin/GAL_ImageNet/GAL_KD2
##python3 main_res_ksefinal.py --teacher_dir ./pytorch_resnet_cifar10-master/pretrained_models/resnet110.th
#CUDA_VISIBLE_DEVICES=1 python3 finetune_resnet.py --refine ./experiment11/checkpoint/model_69.pt
##python3 finetune_resnet.py --refine ./experiment1/checkpoint/model_56.pt