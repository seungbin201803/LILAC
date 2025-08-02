#!/bin/bash
#
#SBATCH --job-name=lilac_rt_order_21-3 # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
##SBATCH --time=48:00:00 # set this time according to your need
#SBATCH --mem=32GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -w ai-gpu05

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=sep4013@med.cornell.edu

module purge
module load anaconda3
source ./venv/bin/activate
source activate lilac2
#Or if in your home dir: source ~/myvenv/bin/activate

# python3 ./run.py \
#     --jobname='lilac_rt_order_21-3' \
#     --task_option='o' \
#     --targetname='timepoint' \
#     --backbone_name='cnn_3D' \
#     --batchsize=8 \
#     --max_epoch=10 \
#     --output_directory='./output' \
#     --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_80' \
#     --image_size='80,80,80' \
#     --csv_file_train='./RT_easy_crop80-ants-sim_train.csv' \
#     --csv_file_val='./RT_easy_crop80-ants-sim_val.csv' \
#     --csv_file_test='./RT_easy_crop80-ants-sim_test.csv' \
#     --earlystopping=100 \
#     --lrscheduler 20 0.5 \
# --inter_num_ch=16 \
# --num_block=6
    
# 710423
# new data
# easy
# no early stop, lrscheduler 20 0.5
# epoch 10


# # ### test
# python3 ./run.py \
#     --jobname='lilac_rt_order_21-3' \
#     --task_option='o' \
#     --targetname='timepoint' \
#     --backbone_name='cnn_3D' \
#     --batchsize=8 \
#     --max_epoch=10 \
#     --output_directory='./output' \
#     --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_80' \
#     --image_size='80,80,80' \
#     --csv_file_train='./RT_easy_crop80-ants-sim_train.csv' \
#     --csv_file_val='./RT_easy_crop80-ants-sim_val.csv' \
#     --csv_file_test='./RT_easy_crop80-ants-sim_test.csv' \
#     --earlystopping=100 \
#     --lrscheduler 20 0.5 \
#     --run_mode='eval'
# # 710802

### gradcam
python3 ./run.py \
    --jobname='lilac_rt_order_21-3' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='cnn_3D' \
    --batchsize=8 \
    --max_epoch=10 \
    --output_directory='./output' \
    --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_80' \
    --image_size='80,80,80' \
    --csv_file_train='./RT_easy_crop80-ants-sim_train.csv' \
    --csv_file_val='./RT_easy_crop80-ants-sim_val.csv' \
    --csv_file_test='./RT_easy_crop80-ants-sim_test.csv' \
    --earlystopping=100 \
    --lrscheduler 20 0.5 \
    --run_mode='eval'\
    --gradcam