#!/bin/bash
#
#SBATCH --job-name=move # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
##SBATCH --time=48:00:00 # set this time according to your need
#SBATCH --mem=32GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -w ai-gpu02

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=sep4013@med.cornell.edu

module purge
module load anaconda3
source ./venv/bin/activate
source activate lilac


python3 ./move_file.py