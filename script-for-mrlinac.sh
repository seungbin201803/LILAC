#/usr/bin/bash


# f1-fl model - train
## Update image_directory to the path where your images are saved.
python3 ./run.py \
    --jobname='f1fl' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='resnet18_3D' \
    --batchsize=8 \
    --max_epoch=100 \
    --output_directory='./output' \
    --image_directory='./image' \
    --image_size='80,80,80' \
    --csv_file_train='./demo_for_release/demo_mrlinac-f1fl_train.csv' \
    --csv_file_val='./demo_for_release/demo_mrlinac-f1fl_val.csv' \
    --csv_file_test='./demo_for_release/demo_mrlinac-f1fl_test.csv' \
    --earlystopping=100 \
    --lrscheduler 20 0.5 \

# f1-fl model - test
## Update image_directory to the path where your images are saved.
python3 ./run.py \
    --run_mode='eval' \
    --jobname='f1fl' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='resnet18_3D' \
    --batchsize=8 \
    --output_directory='./output' \
    --image_directory='./image' \
    --image_size='80,80,80' \
    --csv_file_train='./demo_for_release/demo_mrlinac-f1fl_train.csv' \
    --csv_file_val='./demo_for_release/demo_mrlinac-f1fl_val.csv' \
    --csv_file_test='./demo_for_release/demo_mrlinac-f1fl_test.csv' \

# all-pair model - train, starting from the best f1-fl model
## Update image_directory to the path where your images are saved.
python3 ./run.py \
    --jobname='allpair' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='resnet18_3D' \
    --batchsize=8 \
    --max_epoch=100 \
    --output_directory='./output' \
    --image_directory='/midtier/sablab/scratch/data/Prostate_RadiologyTreatment/fraction_crop_segmentation' \
    --image_size='80,80,80' \
    --csv_file_train='./demo_for_release/demo_mrlinac-allpair_train.csv' \
    --csv_file_val='./demo_for_release/demo_mrlinac-allpair_val.csv' \
    --csv_file_test='./demo_for_release/demo_mrlinac-allpair_test.csv' \
    --earlystopping=100 \
    --lrscheduler 20 0.5 \
    --path_pretrained_model='./output/f1fl-temporal_ordering-backbone_resnet18_3D-lr0.001-seed0-batch8/model_best.pth' \

# all-pair model - test + gradcam
## Update image_directory to the path where your images are saved.
python3 ./run.py \
    --run_mode='eval' \
    --gradcam=True \
    --jobname='allpair' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='resnet18_3D' \
    --batchsize=8 \
    --max_epoch=100 \
    --output_directory='./output' \
    --image_directory='./image' \
    --image_size='80,80,80' \
    --csv_file_train='./demo_for_release/demo_mrlinac-allpair_train.csv' \
    --csv_file_val='./demo_for_release/demo_mrlinac-allpair_val.csv' \
    --csv_file_test='./demo_for_release/demo_mrlinac-allpair_test.csv' \

