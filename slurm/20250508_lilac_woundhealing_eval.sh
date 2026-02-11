#!/bin/bash
#
#SBATCH --job-name=lilac_woundhealing_20250429 # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00 # set this time according to your need
#SBATCH --mem=8GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -w ai-gpu03
module purge
module load anaconda3
#source ./venv/bin/activate
source activate lilac
# Or if in your home dir: source ~/myvenv/bin/activate
python3 ./run.py \
    --jobname='lilac_woundhealing_20250429' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='cnn_2D' \
    --batchsize=128 \
    --max_epoch=40 \
    --output_directory='./output' \
    --image_directory='/home/sep4013/woundheaeling2_single' \
    --image_size='224,224' \
    --csv_file_train='./demo_for_release/demo_woundhealing_train.csv' \
    --csv_file_val='./demo_for_release/demo_woundhealing_val.csv' \
    --csv_file_test='./demo_for_release/demo_woundhealing_test.csv'\
    --run_mode='eval'