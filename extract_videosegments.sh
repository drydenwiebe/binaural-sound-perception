#!/bin/bash
#SBATCH --account=rrg-kyi
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=0-72:00
#SBATCH --output=extract_videosegments_9.log 

cd /home/drydenw/projects/rrg-kyi/drydenw/binaural-sound-perception
python extract_videosegments.py 
