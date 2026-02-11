#!/bin/bash
#
#SBATCH --job-name=age_feature38_allpair_1 # give your job a name
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

python3 ./run.py \
    --jobname='age_feature38_allpair_1' \
    --task_option='s' \
    --targetname='timepoint' \
    --backbone_name='simplemlp' \
    --batchsize=8 \
    --max_epoch=100 \
    --output_directory='./output' \
    --image_directory='/midtier/sablab/scratch/sep4013/LILAC/output/lilac_rt_order_38-temporal_ordering-backbone_resnet18_3D-lr0.001-seed0-batch8/save_feature' \
    --csv_file_train='./RT_age_allpair__train.csv' \
    --csv_file_val='./RT_age_allpair__val.csv' \
    --csv_file_test='./RT_age_allpair__test.csv' \
    --dataloader_type='feature38' \
    --mlp_hidden_size_list 0 \
    --mlp_input_size=64000 \
    --earlystopping=100 \

# 818410