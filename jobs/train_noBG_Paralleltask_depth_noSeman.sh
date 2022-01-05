#!/bin/bash
#SBATCH --account=rrg-kyi
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=0-72:00
#SBATCH --gres=gpu:1 #1 gpu
#SBATCH --output=train_noBG_Paralleltask_depth_noSeman.log  

####################################################################
cd /home/drydenw/projects/rrg-kyi/drydenw/binaural-sound-perception
python train_noBG_Paralleltask_depth_noSeman.py --lr 0.00001581
