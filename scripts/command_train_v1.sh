#/bin/bash
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train1 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train2 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train3 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train4 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train5 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train6 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train7 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train8 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train9 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train10 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity



#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzirgb/train1 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors
