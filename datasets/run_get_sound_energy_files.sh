#!/bin/bash
#SBATCH --account=rrg-kyi
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-8:00
#SBATCH --output=get_sound_energy_files.log 

cd /home/drydenw/projects/rrg-kyi/drydenw/binaural-sound-perception/datasets
python get_sound_energy_files.py
