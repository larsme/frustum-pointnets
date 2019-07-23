#/bin/bash
#47 epochs trained
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/log_v1/train1 --num_point 1024 --max_epoch 53 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --restore_model_path ./train/log_v1/train1/model-ckpt
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/log_v1/train2 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/log_v1/train3 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/log_v1/train4 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/log_v1/train5 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/log_v1/train6 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/log_v1/train7 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/log_v1/train8 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/log_v1/train9 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/log_v1/train10 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
