#!/bin/bash
#SBATCH --account=rrg-kyi
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-8:00
#SBATCH --output=extract_wavsplits_track_8.log 

cd /home/drydenw/projects/rrg-kyi/drydenw/binaural-sound-perception
#source ../bi_sound/bin/activate
python extract_wavsplits.py
