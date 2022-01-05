#!/bin/bash
#SBATCH --account=rrg-kyi
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-8:00
#SBATCH --output=extract_spectrograms_track_3.log 

cd /home/drydenw/projects/rrg-kyi/drydenw/binaural-sound-perception
source ../bi_sound/bin/activate
python extract_spectrograms.py
