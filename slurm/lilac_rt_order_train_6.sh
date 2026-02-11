#!/bin/bash
#
#SBATCH --job-name=lilac_rt_order_6 # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
##SBATCH --time=48:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -w ai-gpu03

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=sep4013@med.cornell.edu

module purge
module load anaconda3
source ./venv/bin/activate
source activate lilac
# Or if in your home dir: source ~/myvenv/bin/activate
# python3 ./run.py \
#     --jobname='lilac_rt_order_6' \
#     --task_option='o' \
#     --targetname='timepoint' \
#     --backbone_name='cnn_3D' \
#     --batchsize=8 \
#     --max_epoch=100 \
#     --output_directory='./output' \
#     --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/image_crop_128' \
#     --image_size='128,128,128' \
#     --csv_file_train='./RT_easy_crop128_1_20250608_train.csv' \
#     --csv_file_val='./RT_easy_crop128_1_20250608_val.csv' \
#     --csv_file_test='./RT_easy_crop128_1_20250608_test.csv' \
#     --inter_num_ch=32
    
# 498948
# inter_num_ch=32
# RT_easy_crop128_1_20250608
# cpus-per-task=12, batchsize=8

python3 ./run.py \
    --jobname='lilac_rt_order_6' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='cnn_3D' \
    --batchsize=8 \
    --max_epoch=100 \
    --output_directory='./output' \
    --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/image_crop_128' \
    --image_size='128,128,128' \
    --csv_file_train='./RT_easy_crop128_1_20250608_train.csv' \
    --csv_file_val='./RT_easy_crop128_1_20250608_val.csv' \
    --csv_file_test='./RT_easy_crop128_1_20250608_test.csv' \
    --inter_num_ch=32 \
    --run_mode='eval'
# 500106