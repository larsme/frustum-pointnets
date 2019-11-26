#/bin/bash

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_guided_completion_no_box_certainty_2048/train1 --num_point 2048 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --avoid_point_duplicates --with_colors
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyz_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --depth_completion_augmentation
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./train/logs/xyz_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --depth_completion_augmentation --with_colors




#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_c_guided_completion_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --with_depth_confidences --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_guided_completion_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_guided_completion_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --avoid_point_duplicates --with_colors

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_c_guided_completion_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --with_depth_confidences --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_guided_completion_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_guided_completion_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --avoid_point_duplicates --with_colors

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_c_guided_completion_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --with_depth_confidences --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_guided_completion_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_guided_completion_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --avoid_point_duplicates --with_colors

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_c_guided_completion_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --with_depth_confidences --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_guided_completion_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_guided_completion_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_guided_depth_completion --avoid_point_duplicates --with_colors


#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_prediction_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_depth_prediction --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_prediction_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_depth_prediction --avoid_point_duplicates --with_colors
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_c_unguided_completion_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --with_depth_confidences --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_unguided_completion_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_unguided_completion_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --avoid_point_duplicates --with_colors

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_prediction_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_depth_prediction --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_prediction_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_depth_prediction --avoid_point_duplicates --with_colors
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_c_unguided_completion_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --with_depth_confidences --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_unguided_completion_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_unguided_completion_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --avoid_point_duplicates --with_colors

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_prediction_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_depth_prediction --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_prediction_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_depth_prediction --avoid_point_duplicates --with_colors
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_c_unguided_completion_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --with_depth_confidences --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_unguided_completion_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_unguided_completion_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --avoid_point_duplicates --with_colors

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_prediction_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_depth_prediction --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_prediction_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_depth_prediction --avoid_point_duplicates --with_colors
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_c_unguided_completion_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --with_depth_confidences --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_unguided_completion_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --avoid_point_duplicates
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_unguided_completion_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --from_unguided_depth_completion --avoid_point_duplicates --with_colors


#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities


#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_colors
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_colors
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_colors
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_colors


#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --dont_input_box_probabilities
#
#
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5



#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./logs/xyzi/train1/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./logs/xyzi/train2/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./logs/xyzi/train3/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./logs/xyzi/train4/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi/train5 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./logs/xyzi/train5/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi/train6 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./logs/xyzi/train6/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi/train7 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./logs/xyzi/train7/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi/train8 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./logs/xyzi/train8/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi/train9 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./logs/xyzi/train9/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi/train10 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --restore_model_path ./logs/xyzi/train10/model.ckpt
##
###/bin/bash
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzirgb/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./logs/xyzirgb/train1/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzirgb/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./logs/xyzirgb/train2/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzirgb/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./logs/xyzirgb/train3/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzirgb/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./logs/xyzirgb/train4/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzirgb/train5 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./logs/xyzirgb/train5/model.ckpt
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzirgb/train6 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --restore_model_path ./logs/xyzirgb/train6/model.ckpt

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzirgb_no_box_certainty/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzirgb_no_box_certainty/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzirgb_no_box_certainty/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --dont_input_box_probabilities
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzirgb_no_box_certainty/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --with_intensity --with_colors --dont_input_box_probabilities



# depth completion augmentation

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi_512/train1 --num_point 512 --max_epoch 100 --batch_size 100 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_intensity
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi_512/train2 --num_point 512 --max_epoch 100 --batch_size 100 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_intensity
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi_512/train3 --num_point 512 --max_epoch 100 --batch_size 100 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_intensity
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzi_512/train4 --num_point 512 --max_epoch 100 --batch_size 100 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_intensity

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --restore_epoch 90 --restore_model_path ./logs/xyz_no_box_certainty_completion_augmentation/train1/model.ckpt --log_dir ./logs/xyz_no_box_certainty_completion_augmentation/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --depth_completion_augmentation
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_no_box_certainty_completion_augmentation/train1 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_colors --depth_completion_augmentation
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_no_box_certainty_completion_augmentation/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --depth_completion_augmentation
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_no_box_certainty_completion_augmentation/train2 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_colors --depth_completion_augmentation
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_no_box_certainty_completion_augmentation/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --depth_completion_augmentation
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_no_box_certainty_completion_augmentation/train3 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_colors --depth_completion_augmentation
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyz_no_box_certainty_completion_augmentation/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --depth_completion_augmentation
#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir ./logs/xyzrgb_no_box_certainty_completion_augmentation/train4 --num_point 1024 --max_epoch 100 --batch_size 50 --decay_step 800000 --decay_rate 0.5 --dont_input_box_probabilities --with_colors --depth_completion_augmentation


