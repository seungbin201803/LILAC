#!/bin/bash
#
#SBATCH --job-name=LOADER_FEATURES_histogram_easy_2.sh # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
##SBATCH --time=48:00:00 # set this time according to your need
#SBATCH --mem=50GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
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

python3 ./run.py \
    --jobname='LOADER_FEATURES_histogram_easy_2' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='simplemlp' \
    --batchsize=8 \
    --max_epoch=100 \
    --output_directory='./output' \
    --csv_file_train='./RT_histogram_bin01_easy_15_train.csv' \
    --csv_file_val='./RT_histogram_bin01_easy_15_val.csv' \
    --csv_file_test='./RT_histogram_bin01_easy_15_test.csv' \
    --earlystopping=100 \
    --mlp_input_size=15 \
    --mlp_hidden_size_list 1 1 1 1 1 \
    --lrscheduler 10 0.5 \

# 781005