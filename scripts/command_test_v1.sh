#/bin/bash
python train/test.py --gpu 0 --num_point 1024 --model frustum_pointnets_v1 --model_path train/pretrained/log_v1/model.ckpt --output train/detection_results_v1 --idx_path kitti/image_sets/val.txt

