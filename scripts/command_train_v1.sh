#/bin/bash
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities


#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --dont_input_box_probabilities
#
#
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyz/train1 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyz/train2 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyz/train3 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyz/train4 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5



#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train1 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./train/logs/xyzi/train1/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train2 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./train/logs/xyzi/train2/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train3 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./train/logs/xyzi/train3/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train4 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./train/logs/xyzi/train4/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train5 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./train/logs/xyzi/train5/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train6 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./train/logs/xyzi/train6/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train7 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./train/logs/xyzi/train7/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train8 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./train/logs/xyzi/train8/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train9 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./train/logs/xyzi/train9/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzi/train10 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./train/logs/xyzi/train10/model.ckpt
##
###/bin/bash
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzirgb/train1 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./train/logs/xyzirgb/train1/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzirgb/train2 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./train/logs/xyzirgb/train2/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzirgb/train3 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./train/logs/xyzirgb/train3/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzirgb/train4 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./train/logs/xyzirgb/train4/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzirgb/train5 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./train/logs/xyzirgb/train5/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyzirgb/train6 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./train/logs/xyzirgb/train6/model.ckpt
