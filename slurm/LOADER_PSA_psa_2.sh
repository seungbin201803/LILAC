#!/bin/bash
#
#SBATCH --job-name=LOADER_PSA_psa_2 # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
##SBATCH --time=48:00:00 # set this time according to your need
#SBATCH --mem=50GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
##SBATCH -w ai-gpu04 #train
#SBATCH -w ai-gpu08


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
#     --jobname='LOADER_PSA_psa_2' \
#     --task_option='o' \
#     --targetname='timepoint' \
#     --backbone_name='resnet18_3D' \
#     --batchsize=8 \
#     --max_epoch=100 \
#     --output_directory='./output' \
#     --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_segmentation' \
#     --image_size='80,80,80' \
#     --csv_file_train='./RT_PSA__train.csv' \
#     --csv_file_val='./RT_PSA__val.csv' \
#     --csv_file_test='./RT_PSA__test.csv' \
#     --earlystopping=100 \
#     --lrscheduler 5 0.8 \
# changed lrscheduler
# 779608

python3 ./run.py \
    --jobname='LOADER_PSA_psa_2' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='resnet18_3D' \
    --batchsize=8 \
    --max_epoch=100 \
    --output_directory='./output' \
    --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_segmentation' \
    --image_size='80,80,80' \
    --csv_file_train='./RT_PSA__train.csv' \
    --csv_file_val='./RT_PSA__val.csv' \
    --csv_file_test='./RT_PSA__test.csv' \
    --earlystopping=100 \
    --lrscheduler 5 0.8 \
    --run_mode='eval'
# 780905
