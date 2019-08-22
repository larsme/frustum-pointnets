#/bin/bash
#python train/test.py --gpu 0 --num_point 1024 --model frustum_pointnets_v1 --model_path train/pretrained/log_v1/model.ckpt --output train/detection_results_v1 --idx_path kitti/image_sets/val.txt

python train/test.py --gpu 0 --model frustum_pointnets_v1 --model_path ./train/logs/xyz_c_guided_completion_no_box_certainty/train1/model.ckpt --num_point 1024 --batch_size 32 --output ./train/logs/xyz_c_guided_completion_no_box_certainty/train1/detection_results_v1 --idx_path kitti/image_sets/val.txt --dont_input_box_probabilities --from_guided_depth_completion --with_depth_confidences --avoid_point_duplicates

