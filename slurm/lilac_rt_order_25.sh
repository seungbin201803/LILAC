#!/bin/bash
#
#SBATCH --job-name=lilac_rt_order_25 # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
##SBATCH --time=48:00:00 # set this time according to your need
#SBATCH --mem=32GB #64GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -w ai-gpu04

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
#     --jobname='lilac_rt_order_25' \
#     --task_option='o' \
#     --targetname='timepoint' \
#     --backbone_name='cnn_3D' \
#     --batchsize=8 \
#     --max_epoch=100 \
#     --output_directory='./output' \
#     --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_80' \
#     --image_size='80,80,80' \
#     --csv_file_train='./RT_allpair_crop80-ants-sim__train.csv' \
#     --csv_file_val='./RT_allpair_crop80-ants-sim__val.csv' \
#     --csv_file_test='./RT_allpair_crop80-ants-sim__test.csv' \
#     --earlystopping=100 \
#     --path_pretrained_model='./output/lilac_rt_order_21-2-temporal_ordering-backbone_cnn_3D-lr0.001-seed0-batch8/model_best.pth' \
#     --lrscheduler 20 0.5 \
# #--inter_num_ch=16 \
# #--num_block=6
    
# 709639
# curriculum learning



# # # ### test
python3 ./run.py \
    --jobname='lilac_rt_order_25' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='cnn_3D' \
    --batchsize=8 \
    --max_epoch=100 \
    --output_directory='./output' \
    --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_80' \
    --image_size='80,80,80' \
    --csv_file_train='./RT_allpair_crop80-ants-sim__train.csv' \
    --csv_file_val='./RT_allpair_crop80-ants-sim__val.csv' \
    --csv_file_test='./RT_allpair_crop80-ants-sim__test.csv' \
    --earlystopping=100 \
    --path_pretrained_model='./output/lilac_rt_order_21-2-temporal_ordering-backbone_cnn_3D-lr0.001-seed0-batch8/model_best.pth' \
    --lrscheduler 20 0.5 \
    --run_mode='eval'
# # 710622