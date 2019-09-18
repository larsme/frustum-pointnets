#/bin/bash

python train/visualize.py --split training --gpu 0 --model frustum_pointnets_v1 --num_point 1024 --dont_input_box_probabilities --with_intensity --restore_model_path ./train/logs/xyzi_no_box_certainty/train1/model.ckpt
python train/visualize.py --split training --gpu 0 --model frustum_pointnets_v1 --num_point 1024 --dont_input_box_probabilities --with_colors  --from_guided_depth_completion --avoid_point_duplicates --restore_model_path ./train/logs/xyzrgb_guided_completion_no_box_certainty/train1/model.ckpt
python train/visualize.py --split training --gpu 0 --model frustum_pointnets_v1 --num_point 1024 --dont_input_box_probabilities --with_colors  --from_unguided_depth_completion --avoid_point_duplicates --restore_model_path ./train/logs/xyzrgb_unguided_completion_no_box_certainty/train1/model.ckpt
python train/visualize.py --split training --gpu 0 --model frustum_pointnets_v1 --num_point 1024 --dont_input_box_probabilities  --from_depth_prediction --avoid_point_duplicates --restore_model_path ./train/logs/xyz_prediction_no_box_certainty/train1/model.ckpt

python train/visualize.py --from_rgb_detection --split training --gpu 0 --model frustum_pointnets_v1 --num_point 1024 --dont_input_box_probabilities --with_intensity --restore_model_path ./train/logs/xyzi_no_box_certainty/train1/model.ckpt
python train/visualize.py --from_rgb_detection --split training --gpu 0 --model frustum_pointnets_v1 --num_point 1024 --dont_input_box_probabilities --with_colors  --from_guided_depth_completion --avoid_point_duplicates --restore_model_path ./train/logs/xyzrgb_guided_completion_no_box_certainty/train1/model.ckpt
python train/visualize.py --from_rgb_detection --split training --gpu 0 --model frustum_pointnets_v1 --num_point 1024 --dont_input_box_probabilities --with_colors  --from_unguided_depth_completion --avoid_point_duplicates --restore_model_path ./train/logs/xyzrgb_unguided_completion_no_box_certainty/train1/model.ckpt
python train/visualize.py --from_rgb_detection --split training --gpu 0 --model frustum_pointnets_v1 --num_point 1024 --dont_input_box_probabilities  --from_depth_prediction --avoid_point_duplicates --restore_model_path ./train/logs/xyz_prediction_no_box_certainty/train1/model.ckpt