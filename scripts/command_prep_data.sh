#/bin/bash
python kitti/prepare_data.py --from_depth_completion --fill_n_points=2048 --gen_train_rgb_detection  #--gen_val_rgb_detection # --gen_train --gen_val #--show_pixel_statistics
