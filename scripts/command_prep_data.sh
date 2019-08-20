#/bin/bash
#python kitti/prepare_data.py --from_guided_depth_completion --fill_n_points=1024 --gen_train_rgb_detection  #--gen_val_rgb_detection # --gen_train --gen_val #--show_pixel_statistics
python kitti/prepare_data.py --from_guided_depth_completion --gen_val_rgb_detection --show_alt_depth_source_seg_statistics
