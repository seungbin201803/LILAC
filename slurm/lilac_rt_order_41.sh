#!/bin/bash
#
#SBATCH --job-name=lilac_rt_order_41 # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
##SBATCH --time=48:00:00 # set this time according to your need
#SBATCH --mem=50GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -w ai-gpu04 #train

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
#     --jobname='lilac_rt_order_41' \
#     --task_option='o' \
#     --targetname='timepoint' \
#     --backbone_name='resnet18_3D' \
#     --batchsize=8 \
#     --max_epoch=100 \
#     --output_directory='./output' \
#     --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/crop_80_BladderXdil2/crop_80' \
#     --image_size='80,80,80' \
#     --csv_file_train='./RT_easy_crop80-ants-sim_train.csv' \
#     --csv_file_val='./RT_easy_crop80-ants-sim_val.csv' \
#     --csv_file_test='./RT_easy_crop80-ants-sim_test.csv' \
#     --earlystopping=100 \
#     --lrscheduler 20 0.5 \
    

#resnet18_3D
#max epoch 100
#easy
#BladderX
#759505, 762300

### test
python3 ./run.py \
    --jobname='lilac_rt_order_41' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='resnet18_3D' \
    --batchsize=8 \
    --max_epoch=100 \
    --output_directory='./output' \
    --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/crop_80_BladderXdil2/crop_80' \
    --image_size='80,80,80' \
    --csv_file_train='./RT_easy_crop80-ants-sim_train.csv' \
    --csv_file_val='./RT_easy_crop80-ants-sim_val.csv' \
    --csv_file_test='./RT_easy_crop80-ants-sim_test.csv' \
    --earlystopping=100 \
    --lrscheduler 20 0.5 \
    --run_mode='eval' \
# 763424
