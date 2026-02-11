#!/bin/bash
#
#SBATCH --job-name=lilac_rt_order_38 # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
##SBATCH --time=48:00:00 # set this time according to your need
#SBATCH --mem=50GB # how much RAM will your notebook consume? 
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
source activate lilac2
#Or if in your home dir: source ~/myvenv/bin/activate

# python3 ./run.py \
#     --jobname='lilac_rt_order_38' \
#     --task_option='o' \
#     --targetname='timepoint' \
#     --backbone_name='resnet18_3D' \
#     --batchsize=8 \
#     --max_epoch=100 \
#     --output_directory='./output' \
#     --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_segmentation' \
#     --image_size='80,80,80' \
#     --csv_file_train='./RT_allpair_crop80-ants-sim__train.csv' \
#     --csv_file_val='./RT_allpair_crop80-ants-sim__val.csv' \
#     --csv_file_test='./RT_allpair_crop80-ants-sim__test.csv' \
#     --earlystopping=100 \
#     --path_pretrained_model='./output/lilac_rt_order_31-temporal_ordering-backbone_resnet18_3D-lr0.001-seed0-batch8/model_best.pth' \
#     --lrscheduler 20 0.5 \
# #--inter_num_ch=16 \
# #--num_block=6
    
# 721475
# crop_segmentation
# Resnet183D
# curriculum learning from epoch100

# python3 ./run.py \
#     --jobname='lilac_rt_order_38' \
#     --task_option='o' \
#     --targetname='timepoint' \
#     --backbone_name='resnet18_3D' \
#     --batchsize=8 \
#     --max_epoch=100 \
#     --output_directory='./output' \
#     --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_segmentation' \
#     --image_size='80,80,80' \
#     --csv_file_train='./RT_allpair_crop80-ants-sim__train.csv' \
#     --csv_file_val='./RT_allpair_crop80-ants-sim__val.csv' \
#     --csv_file_test='./RT_allpair_crop80-ants-sim__test.csv' \
#     --earlystopping=100 \
#     --path_pretrained_model='./output/lilac_rt_order_31-temporal_ordering-backbone_resnet18_3D-lr0.001-seed0-batch8/model_best.pth' \
#     --lrscheduler 20 0.5 \
#     --run_mode='eval' \
# # 721602


# python3 ./run.py \
#     --jobname='lilac_rt_order_38' \
#     --task_option='o' \
#     --targetname='timepoint' \
#     --backbone_name='resnet18_3D' \
#     --batchsize=8 \
#     --max_epoch=100 \
#     --output_directory='./output' \
#     --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_segmentation' \
#     --image_size='80,80,80' \
#     --csv_file_train='./RT_allpair_crop80-ants-sim__train.csv' \
#     --csv_file_val='./RT_allpair_crop80-ants-sim__val.csv' \
#     --csv_file_test='./RT_allpair_crop80-ants-sim__test.csv' \
#     --earlystopping=100 \
#     --lrscheduler 20 0.5 \
#     --run_mode='eval' \
#     --gradcam
# 721608, 775913


# # # test, guideone
# python3 ./run.py \
#     --jobname='lilac_rt_order_38' \
#     --task_option='o' \
#     --targetname='timepoint' \
#     --backbone_name='resnet18_3D' \
#     --batchsize=8 \
#     --max_epoch=100 \
#     --output_directory='./output' \
#     --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_segmentation' \
#     --image_size='80,80,80' \
#     --csv_file_train='./RT_allpair_crop80-ants-sim__train.csv' \
#     --csv_file_val='./RT_allpair_crop80-ants-sim__val.csv' \
#     --csv_file_test='./RT_guideone_crop80-ants-sim_test.csv' \
#     --earlystopping=100 \
#     --lrscheduler 20 0.5 \
#     --run_mode='eval'
# # 732801

### Save features
python3 ./run.py \
    --jobname='lilac_rt_order_38' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='resnet18_3D' \
    --batchsize=8 \
    --max_epoch=100 \
    --output_directory='./output' \
    --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_segmentation' \
    --image_size='80,80,80' \
    --csv_file_train='./RT_allpair_crop80-ants-sim__train.csv' \
    --csv_file_val='./RT_allpair_crop80-ants-sim__val.csv' \
    --csv_file_test='./RT_allpair_crop80-ants-sim__test.csv' \
    --run_mode='savefeature' \
