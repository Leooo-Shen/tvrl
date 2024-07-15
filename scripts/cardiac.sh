#!/bin/bash
# CVRL
python pretrain.py --config-name=pretrain pretrain_framework=simclr data=cardiac_2c8f2s_ssl use_time_embed=False 

# LTN
python pretrain.py --config-name=pretrain pretrain_framework=simclr data=cardiac_2c8f2s_ssl use_time_embed=False use_ltn=True 

# VideoMAE
python pretrain.py --config-name=pretrain pretrain_framework=mae data=cardiac_1c8f2s_ssl use_time_embed=False 

# cSimCLR
python pretrain.py --config-name=pretrain pretrain_framework=simclr data=cardiac_1c8f2s_ssl use_time_embed=False 

# cSimCLR-TE
python pretrain.py --config-name=pretrain pretrain_framework=simclr data=cardiac_1c8f2s_ssl use_time_embed=True

# TVRL
python pretrain.py --config-name=pretrain pretrain_framework=mlm data=cardiac_1c8f2s_ssl use_time_embed=False mlm.mask_ratio=0.15


################
# Fine-tuning  #
################

# supervised
python finetune.py freeze_backbone=False data=cardiac_1c8f2s_sup

# kinetics pretrained model
python finetune_baseline_kinetics.py data=cardiac_1c8f2s_sup_k

# other pretrained checkpoints
pretrained_weights=/your/checkpoint/path/last.ckpt
python finetune.py pretrained_weights=$pretrained_weights data=cardiac_1c8f2s_sup

# for those methods with time embedding, please add use_time_embed=True
python finetune.py pretrained_weights=$pretrained_weights data=cardiac_1c8f2s_sup use_time_embed=True

