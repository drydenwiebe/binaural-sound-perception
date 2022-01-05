#!/bin/bash
#SBATCH --account=rrg-kyi
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-72:00
#SBATCH --gres=gpu:1 #1 gpu
#SBATCH --output=spec2disp.log  

####################################################################
cd /home/drydenw/projects/rrg-kyi/drydenw/binaural-sound-perception
python s2d.py --iter_for_report 10
