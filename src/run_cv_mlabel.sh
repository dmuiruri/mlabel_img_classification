#!/bin/bash
#SBATCH -n 2
#SBATCH -p gputest
#SBATCH -t 00:10:00
#SBATCH --mem=3000
#SBATCH --gres=gpu:v100:1
#SBATCH -J cv_mlabel
#SBATCH -o cv.out.%j
#SBATCH -e cv.err.%j
#SBATCH --account=project_2002605
#SBATCH

module purge
module load python-data/3.9-22.04
module load pytorch

python multi_label_classifier.py
