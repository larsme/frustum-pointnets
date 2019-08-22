''' Training Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image
import matplotlib as plt
import matplotlib.cm
import cv2
import warnings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR in sys.path:
    sys.path.remove(BASE_DIR)
sys.path.append(ROOT_DIR)
from kitti.prepare_data import get_lidar_in_image_fov, read_box_file, extract_pc_in_box3d
from kitti.kitti_object import kitti_object
import kitti.kitti_util as utils
from models.model_util import g_type2class, g_class2type, g_type2onehotclass
from train.provider import rotate_pc_along_y


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1',
                    help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--with_intensity', action='store_true',
                    help='Use Intensity for training')
parser.add_argument('--with_colors', action='store_true',
                    help='Use Colors for training')
parser.add_argument('--with_depth_confidences', action='store_true',
                    help='Use depth completion confidences')
parser.add_argument('--dont_rotate_frustum', action='store_true',
                    help='Dont rotate frutum to center')
parser.add_argument('--from_guided_depth_completion', action='store_true',
                    help='Use point cloud from depth completion')
parser.add_argument('--from_unguided_depth_completion', action='store_true',
                    help='Use point cloud from unguided depth completion')
parser.add_argument('--from_depth_prediction', action='store_true',
                    help='Use point cloud from depth prediction')
parser.add_argument('--restore_model_path', default=None,
                    help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--dont_input_box_probabilities', action='store_true',
                    help='Use box probabilities as net inputs')
parser.add_argument('--split', default='training',
                    help='KITTI Object Dataset Split')
parser.add_argument('--num_point', type=int, default=2048,
                    help='Point Number [default: 2048]')
parser.add_argument('--avoid_point_duplicates', action='store_true',
                    help='Try to avoid point duplicates when sampling')
parser.add_argument('--from_rgb_detection', action='store_true',
                    help='Use boxes from rgb detection')
FLAGS = parser.parse_args()

# Set training configurations
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
NUM_CHANNEL = 3 # point feature channel
if FLAGS.with_intensity:
    NUM_CHANNEL = NUM_CHANNEL + 1
if FLAGS.with_depth_confidences:
    NUM_CHANNEL = NUM_CHANNEL + 1
if FLAGS.with_colors:
    NUM_CHANNEL = NUM_CHANNEL + 3
NUM_CLASSES = 2 # segmentation net has two classes

NUM_REAL_CLASSES = 3
REAL_CLASSES = ['Car', 'Pedestrian', 'Cyclist']

from_depth_completion = FLAGS.from_unguided_depth_completion or FLAGS.from_guided_depth_completion
use_depth_net = from_depth_completion or FLAGS.from_depth_prediction
assert int(FLAGS.from_guided_depth_completion) + int(FLAGS.from_unguided_depth_completion) \
       + int(FLAGS.from_depth_prediction) <= 1

if from_depth_completion:
    if FLAGS.from_guided_depth_completion:
        bla = 0
        # depth_net = load_net('exp_guided_nconv_cnn_l1', mode='bla', checkpoint_num=40, set_='bla')
    else:  # from_unguided_depth_completion:
        sys.path.append(os.path.join(ROOT_DIR, '../nconv'))
        from run_nconv_cnn import load_net

        depth_net = load_net(training_ws_path='workspace/exp_unguided_depth',
                             network_file='network_exp_unguided_depth',
                             params_sub_dir='Default', mode='bla', checkpoint_num=3, sets=None)
    desired_image_height = 352
    desired_image_width = 1216
elif FLAGS.from_depth_prediction:
    sys.path.append(os.path.join(ROOT_DIR, '../monodepth2'))
    from monodepth_external import load_net

    depth_net = load_net("mono+stereo_1024x320", use_cuda=True)

dataset = kitti_object(os.path.join(ROOT_DIR, './../../data/kitti_object'), FLAGS.split)

def get_image_data(FLAGS, image_index, img_height_threshold):
    # image data
    calib = dataset.get_calibration(image_index)  # 3 by 4 matrix

    pc_velo = dataset.get_lidar(image_index)
    pc_rect = np.zeros_like(pc_velo)
    pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
    pc_rect[:, 3] = pc_velo[:, 3]

    img = dataset.get_image(image_index)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape

    _, pts_image_2d, img_fov_inds, pc_image_depths = get_lidar_in_image_fov( \
        pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
    pc_rect = pc_rect[img_fov_inds, :]

    pts_image_2d = np.ndarray.astype(np.round(pts_image_2d[img_fov_inds, :]), int)
    pts_image_2d[pts_image_2d < 0] = 0
    pts_image_2d[pts_image_2d[:, 0] >= img_width, 0] = img_width - 1
    pts_image_2d[pts_image_2d[:, 1] >= img_height, 1] = img_height - 1
    pc_image_depths = pc_image_depths[img_fov_inds]

    # predicted depths
    if FLAGS.from_unguided_depth_completion:
        lidarmap = dataset.generate_depth_map(image_index, 2, desired_image_width, desired_image_height)
        rgb = Image.fromarray(img).resize((desired_image_width, desired_image_height), Image.LANCZOS)
        rgb = np.array(rgb, dtype=np.float16)

        # lidarmap = np.zeros((img_height, img_width), np.float16)
        # for i in range(pc_image_depths.shape[0]):
        #     px = min(max(0, int(round(pts_image_2d[i, 0]))), img_width-1)
        #     py = min(max(0, int(round(pts_image_2d[i, 1]))), img_height-1)
        #     depth = pc_image_depths[i]
        #     if lidarmap[py, px] == 0 or lidarmap[py, px] > depth:
        #         # for conflicts, use closer point
        #         lidarmap[py, px] = depth
        #         # lidarmap[py, px, 2] = 1 # mask
        #         # lidarmap[py, px, 1] = pc_velo[i, 3]
        #         # lidarmap[py, px, 2] = times[i]
        dense_depths, confidences = depth_net.return_one_prediction(lidarmap * 255, rgb, img_width, img_height)
    elif FLAGS.from_guided_depth_completion:
        res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/completed_depth')
        (dense_depths, confidences) = np.load(os.path.join(res_dir, str(image_index) + '.npy'))

        # import matplotlib.pyplot as plt
        # cmap = plt.cm.get_cmap('nipy_spectral', 256)
        # cmap = np.ndarray.astype(np.array([cmap(i) for i in range(256)])[:, :3] * 255, np.uint8)
        #
        # q1_lidar = np.quantile(dense_depths[dense_depths > 0], 0.05)
        # q2_lidar = np.quantile(dense_depths[dense_depths > 0], 0.95)
        # depth_img = cmap[
        #             np.ndarray.astype(np.interp(dense_depths, (q1_lidar, q2_lidar), (0, 255)), np.int_),
        #             :]  # depths
        # fig = Image.fromarray(depth_img)
        # fig.save('depth_img_computed', 'png')
        # fig.show('depth_img_computed')
        #
        # fig = Image.fromarray(img)
        # fig.save('img', 'png')
        # fig.show('img')
        # input()
    elif FLAGS.from_depth_prediction:
        dense_depths = depth_net.return_one_prediction(img, post_process=False)
        confidences = None
    else:
        dense_depths = None
        confidences = None

    # detections
    label_objects = dataset.get_label_objects(image_index)
    pc_labels = np.zeros((np.size(pc_rect, 0)), np.object)
    for label_object in label_objects:
        _, box3d_pts_3d = utils.compute_box_3d(label_object, calib.P)
        _, instance_pc_indexes = extract_pc_in_box3d(pc_rect, box3d_pts_3d)
        overlapping_3d_boxes = np.nonzero(pc_labels[instance_pc_indexes])[0]
        pc_labels[instance_pc_indexes] = label_object.type
        (pc_labels[instance_pc_indexes])[overlapping_3d_boxes] = 'DontCare'

    det_box_class_list = []
    det_box_geometry_list = []
    det_box_certainty_list = []
    if FLAGS.from_rgb_detection:
        for rgb_det_filename in (os.path.join(ROOT_DIR, 'kitti/rgb_detections/rgb_detection_train.txt'),
                                 os.path.join(ROOT_DIR, 'kitti/rgb_detections/rgb_detection_val.txt')):
            all_det_box_image_index_list, all_det_box_class_list, all_det_box_geometry_list, all_det_box_certainty_list = \
                read_box_file(rgb_det_filename)
            for box_idx in range(len(all_det_box_class_list)):
                xmin, ymin, xmax, ymax = all_det_box_geometry_list[box_idx]
                if all_det_box_image_index_list[box_idx] == image_index \
                    and all_det_box_class_list[box_idx] in REAL_CLASSES \
                    and ymax - ymin >= img_height_threshold:
                    det_box_class_list.append(all_det_box_class_list[box_idx])
                    det_box_geometry_list.append(all_det_box_geometry_list[box_idx])
                    det_box_certainty_list.append(all_det_box_certainty_list[box_idx])
    else:
        for label_object in label_objects:
            xmin, ymin, xmax, ymax = label_object.box2d
            if label_object.type in REAL_CLASSES and ymax - ymin >= img_height_threshold:
                det_box_geometry_list.append(label_object.box2d)
                det_box_certainty_list.append(1)
                det_box_class_list.append(label_object.type)

    return img, img_width, img_height, calib, label_objects, \
           pc_rect, pts_image_2d, pc_labels, \
           dense_depths, confidences, \
           det_box_class_list, det_box_geometry_list, det_box_certainty_list


def get_point_cloud(det_box_geometry, det_box_class,
                    pts_image_2d, pc_labels,
                    image_box_detected_labels,
                    lidar_point_threshold,
                    img, calib, label_objects, dense_depths, confidences):
    if det_box_class not in REAL_CLASSES:
        return [], 0, [], [], image_box_detected_labels

    # 2D BOX: Get pts rect backprojected
    xmin, ymin, xmax, ymax = det_box_geometry

    box_fov_inds = (pts_image_2d[:, 0] < xmax) & \
                   (pts_image_2d[:, 0] >= xmin) & \
                   (pts_image_2d[:, 1] < ymax) & \
                   (pts_image_2d[:, 1] >= ymin)
    pc_in_box_count = np.count_nonzero(box_fov_inds)
    # Pass objects that are too small
    if pc_in_box_count < lidar_point_threshold:
        return [], 0, [], [], [], image_box_detected_labels

    pts_2d = pts_image_2d[box_fov_inds, :]
    image_box_detected_labels[box_fov_inds] = True


    # Get frustum angle (according to center pixel in 2D BOX)
    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    uvdepth = np.zeros((1, 3))
    uvdepth[0, 0:2] = box2d_center
    uvdepth[0, 2] = 20  # some random depth
    box2d_center_rect = calib.project_image_to_rect(uvdepth)
    frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                    box2d_center_rect[0, 0])

    if not use_depth_net:
        pc_in_box_colors = img[pts_2d[:, 1], pts_2d[:, 0], :]
        pc_in_box = np.concatenate((pc_rect[box_fov_inds, :], pc_in_box_colors), axis=1)

        pc_in_box_labels = np.zeros((pc_in_box_count), np.int_)
        pc_in_box_labels[pc_labels[box_fov_inds] == det_box_class] = 1
        pc_in_box_labels[pc_labels[box_fov_inds] == 'DontCare'] = -1

        pts_2d_in_box = pts_2d

        pc_in_box_custom_labels = pc_in_box_labels
    else:
        int_x_min = int(max(np.floor(xmin), 0))
        int_x_max = int(min(np.ceil(xmax), img_width - 1))
        box_x_width = int_x_max - int_x_min + 1

        int_y_min = int(max(np.floor(ymin), 0))
        int_y_max = int(min(np.ceil(ymax), img_height - 1))
        box_y_width = int_y_max - int_y_min + 1

        box_sub_pixels_row, box_sub_pixels_col = np.indices((box_y_width, box_x_width))
        box_sub_pixels_row = np.reshape(box_sub_pixels_row, -1)
        box_sub_pixels_col = np.reshape(box_sub_pixels_col, -1)
        pixels_in_box_row = box_sub_pixels_row + int_y_min
        pixels_in_box_col = box_sub_pixels_col + int_x_min

        labels = np.zeros((box_y_width, box_x_width), np.int_) - 1
        true_inds = np.squeeze(pc_labels[box_fov_inds] == det_box_class)
        false_inds = np.logical_and(np.logical_not(true_inds),
                                    np.squeeze(pc_labels[box_fov_inds] != 'DontCare'))
        labels[pts_2d[true_inds, 1] - int_y_min, pts_2d[true_inds, 0] - int_x_min] = 1
        labels[pts_2d[false_inds, 1] - int_y_min, pts_2d[false_inds, 0] - int_x_min] = 0

        pc_in_box_labels = labels.reshape(-1)

        depths_in_box = dense_depths[pixels_in_box_row, pixels_in_box_col]
        new_pc_img_in_box = np.concatenate((np.ndarray.astype(np.expand_dims(pixels_in_box_col, 1), np.float),
                                            np.ndarray.astype(np.expand_dims(pixels_in_box_row, 1), np.float),
                                            np.expand_dims(depths_in_box, 1)), axis=1)
        new_pc_rect_in_box = calib.project_image_to_rect(new_pc_img_in_box)
        pc_in_box_colors = img[pixels_in_box_row, pixels_in_box_col, :]

        if from_depth_completion:
            confidences_in_box = np.expand_dims(confidences[pixels_in_box_row, pixels_in_box_col], 1)
            pc_in_box = np.concatenate((new_pc_rect_in_box, confidences_in_box, pc_in_box_colors), axis=1)
        else:  # from_depth_prediction
            pc_in_box = np.concatenate((new_pc_rect_in_box, pc_in_box_colors), axis=1)
        pts_2d_in_box = np.concatenate((np.expand_dims(pixels_in_box_col, 1),
                                        np.expand_dims(pixels_in_box_row, 1)), axis=1)

        pc_in_box_custom_labels = np.zeros((np.size(new_pc_rect_in_box, 0)), np.int)
        for label_object in label_objects:
            _, box3d_pts_3d = utils.compute_box_3d(label_object, calib.P)
            _, instance_pc_indexes = extract_pc_in_box3d(new_pc_rect_in_box, box3d_pts_3d)
            overlapping_3d_boxes = np.nonzero(pc_in_box_custom_labels[instance_pc_indexes])[0]
            pc_in_box_custom_labels[instance_pc_indexes] = 1
            (pc_in_box_custom_labels[instance_pc_indexes])[overlapping_3d_boxes] = -1

    return pc_in_box, frustum_angle, pc_in_box_labels, pc_in_box_custom_labels, pts_2d_in_box, image_box_detected_labels


def generate_seg_inputs(pc_in_box, frustum_angle, det_box_class, det_box_certainty, gt_pc_labels):

    if FLAGS.from_depth_prediction:
        channels = np.zeros(6, np.bool_)
    elif from_depth_completion:
        channels = np.zeros(7, np.bool_)
    else:
        channels = np.zeros(7, np.bool_)
    channels[:3] = 1
    i_channel = 3
    if not use_depth_net:
        if FLAGS.with_intensity:
            channels[i_channel] = 1
        i_channel += 1
    elif from_depth_completion:
        if FLAGS.with_depth_confidences:
            channels[i_channel] = 1
        i_channel += 1
    if FLAGS.with_colors:
        channels[i_channel:] = 1
    i_channel += 3

    input_pc = pc_in_box[:, channels]

    if not FLAGS.dont_rotate_frustum:
        input_pc = rotate_pc_along_y(input_pc, np.pi/2.0 + frustum_angle)

    # Compute one hot vector
    box_class_one_hot_vec = np.zeros((1, NUM_REAL_CLASSES), np.bool_)
    box_class_one_hot_vec[0, g_type2onehotclass[det_box_class]] = 1

    return input_pc, box_class_one_hot_vec, np.expand_dims(det_box_certainty, 0), gt_pc_labels != -1


def segment_box(sess, ops, input_pc, box_class_one_hot_vec, box_certainty, box_icare):
    left_to_sample = range(input_pc.shape[0])
    pred_labels = np.zeros(input_pc.shape[0], np.int_)-1


    while len(left_to_sample)>0:
        if len(left_to_sample) <= NUM_POINT:
            choice = np.zeros(NUM_POINT, np.int_)
            choice[0:len(left_to_sample)] = left_to_sample
            if FLAGS.avoid_point_duplicates and input_pc.shape[0] >= NUM_POINT:
                sampled = np.delete(range(input_pc.shape[0]), left_to_sample)
                choice[len(left_to_sample):NUM_POINT] = np.random.choice(sampled,
                                                                         NUM_POINT - len(left_to_sample),
                                                                         replace=False)
            else:
                choice[len(left_to_sample):NUM_POINT] = np.random.choice(input_pc.shape[0],
                                                                         NUM_POINT - len(left_to_sample),
                                                                         replace=True)
            permut = np.random.permutation(range(NUM_POINT))
            choice = choice[permut]
            left_to_sample = []
        else:
            choice = np.random.permutation(left_to_sample)
            left_to_sample = choice[NUM_POINT:]
            choice = choice[:NUM_POINT]
        box_input_pc = np.expand_dims(input_pc[choice, :], 0)

        feed_dict = {ops['input_pc']: box_input_pc,
                     ops['box_certainty']: box_certainty,
                     ops['box_class_one_hot_vec']: box_class_one_hot_vec}

        [pc_pred_logits] = \
            sess.run([ops['pc_pred_logits']],
                     feed_dict=feed_dict)
        pred_labels[choice[box_icare[choice]]] \
            = np.argmax(pc_pred_logits, 2).squeeze()[box_icare[choice]]
        box_icare[choice] = False
    return pred_labels


def segment_image_point_cloud(input_pc_list, frustum_angle_list, det_box_class_list, det_box_certainty_list,
                              gt_pc_labels):

    box_pred_label_list = []
    completed_box_pred_label_list = []

    ''' Main function for training and simple evaluation. '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):

            with tf.name_scope("batch_input"):
                input_pc, box_certainty, _, box_class_one_hot_vec = \
                    MODEL.placeholder_inputs(1, NUM_POINT, NUM_CHANNEL, NUM_REAL_CLASSES)

                if FLAGS.dont_input_box_probabilities:
                    batch_box_label_prob = tf.to_float(box_class_one_hot_vec, 'batch_one_hot_vec')
                else:
                    batch_box_label_prob = tf.multiply(tf.to_float(box_class_one_hot_vec),
                                                       tf.tile(tf.expand_dims(box_certainty, axis=1),
                                                               [1, NUM_REAL_CLASSES]),
                                                       'batch_box_label_prob')
            # Get model and losses
            # end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
            #     is_training_pl, bn_decay=bn_decay)

            # logits for element or no element (not just 1 prob)
            pc_pred_logits = MODEL.get_model(input_pc, batch_box_label_prob, tf.constant(False, dtype=tf.bool),
                                             bn_decay=0)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        if FLAGS.restore_model_path is None:
            warnings.warn('-restore_model_path is None; model will be initialized randomly')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)
        ops = {'input_pc': input_pc,
               'box_certainty': box_certainty,
               'box_class_one_hot_vec': box_class_one_hot_vec,
               'pc_pred_logits': pc_pred_logits}

        for box_i in range(len(det_box_class_list)):
            input_pc, box_class_one_hot_vec, box_class_certainty, box_icare = \
                generate_seg_inputs(input_pc_list[box_i], frustum_angle_list[box_i], det_box_class_list[box_i],
                                    det_box_certainty_list[box_i], gt_pc_labels[box_i])
            box_pred_labels = segment_box(sess, ops, input_pc, box_class_one_hot_vec, box_class_certainty, box_icare)
            box_pred_label_list.append(box_pred_labels)
            if use_depth_net:
                completed_box_pred_labels = segment_box(sess, ops, input_pc, box_class_one_hot_vec, box_class_certainty,
                                                      np.ones_like(box_icare))
                completed_box_pred_label_list.append(completed_box_pred_labels)

    return box_pred_label_list, completed_box_pred_label_list


def class_to_color(type):
    if type not in REAL_CLASSES:
        return 0, 0, 0
    else:
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.ndarray.astype(np.array([cmap(i) for i in range(256)])[:, :3]*255, np.int_)
        c = cmap[int(REAL_CLASSES.index(type)*255/NUM_REAL_CLASSES)]
        c2 = tuple((int(c[0]), int(c[1]), int(c[2])))
        return c2


def show_image_with_2d_boxes(img, det_box_geometry_list, det_box_class_list):
    ''' Show image with 2D bounding boxes '''
    img2 = np.copy(img) # for 2d bbox
    for box_i in range(len(det_box_geometry_list)):
        xmin, ymin, xmax, ymax = det_box_geometry_list[box_i]
        cv2.rectangle(img2, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)),
                      color=class_to_color(det_box_class_list[box_i]), thickness=2)
    fig = Image.fromarray(img2)
    fig.show('img_with_2d_boxes')
    fig.save('img_with_2d_boxes.png')


def show_image_with_3d_boxes(img, objects, calib):
    ''' Show image with 2D bounding boxes '''
    img2 = np.copy(img) # for 3d bbox
    for obj in objects:
        if obj.type in REAL_CLASSES:
            box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=class_to_color(obj.type), thickness=2)
    fig = Image.fromarray(img2)
    fig.show('img_with_3d_boxes')
    fig.save('img_with_3d_boxes.png')


def show_segmented_image(img, pts_2d_in_box_list, pc_in_box_labels_list, box_class_list, title):
    label_img = np.zeros_like(img)
    labeled = np.zeros((img.shape[0], img.shape[1]), np.int_)
    for box_i in range(len(box_class_list)):
        box_color = class_to_color(box_class_list[box_i])
        pts_2d_in_box = np.array(pts_2d_in_box_list[box_i], dtype=np.int_)
        pc_in_box_labels = pc_in_box_labels_list[box_i]

        labels = labeled[pts_2d_in_box[:, 1], pts_2d_in_box[:, 0]]

        true_was_true = np.logical_and(pc_in_box_labels == 1, labels == 1)
        true_was_not_true = np.logical_and(pc_in_box_labels == 1, labels != 1)

        false_was_not_true = np.logical_and(pc_in_box_labels == 0, labels != 1)

        dontcare = np.logical_and(pc_in_box_labels == -1, labels == -1)

        px_true_was_not_true = pts_2d_in_box[true_was_not_true, 0]
        py_true_was_not_true = pts_2d_in_box[true_was_not_true, 1]

        px_true_was_true = pts_2d_in_box[true_was_true, 0]
        py_true_was_true = pts_2d_in_box[true_was_true, 1]

        px_false_was_not_true = pts_2d_in_box[false_was_not_true, 0]
        py_false_was_not_true = pts_2d_in_box[false_was_not_true, 1]

        px_dont_care = pts_2d_in_box[dontcare, 0]
        py_dont_care = pts_2d_in_box[dontcare, 1]

        label_img[py_true_was_not_true, px_true_was_not_true] = box_color
        label_img[py_false_was_not_true, px_false_was_not_true] = (255, 255, 255)
        label_img[py_dont_care, px_dont_care] = (60, 60, 60)
        label_img[py_true_was_true, px_true_was_true] = (60, 60, 60)

        labeled[py_true_was_not_true, px_true_was_not_true] = 1
        labeled[py_false_was_not_true, px_false_was_not_true] = 0
    label_img = np.ndarray.astype(label_img/2+img/2, np.uint8)

    fig = Image.fromarray(label_img)
    fig.save(title+'.png')
    fig.show(title)


print('--- Loading Model ---')
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

img_height_threshold = 25
lidar_point_threshold = 5

while True:
    image_index = int(input('image index: '))
    img, img_width, img_height, calib, label_objects, \
               pc_rect, pts_image_2d, pc_labels, \
               dense_depths, confidences, \
               det_box_class_list1, det_box_geometry_list1, det_box_certainty_list1 \
               = get_image_data(FLAGS, image_index, img_height_threshold)
    image_box_detected_labels = np.zeros(pc_labels.shape[0], np.bool_)

    pc_in_box_list = []
    frustum_angle_list = []
    pts_2d_in_box_list = []
    pc_in_box_labels_list = []
    pc_in_box_custom_labels_list = []
    det_box_class_list = []
    det_box_geometry_list = []
    det_box_certainty_list = []
    for box_i in range(len(det_box_geometry_list1)):
        pc_in_box, frustum_angle, pc_in_box_labels, pc_in_box_custom_labels, pts_2d_in_box, image_box_detected_labels\
            = get_point_cloud(det_box_geometry_list1[box_i], det_box_class_list1[box_i],
                              pts_image_2d, pc_labels,
                              image_box_detected_labels,
                              lidar_point_threshold,
                              img, calib, label_objects, dense_depths, confidences)
        if len(pc_in_box_labels) > lidar_point_threshold:
            pc_in_box_list.append(pc_in_box)
            frustum_angle_list.append(frustum_angle)
            pts_2d_in_box_list.append(pts_2d_in_box)
            pc_in_box_labels_list.append(pc_in_box_labels)
            pc_in_box_custom_labels_list.append(pc_in_box_custom_labels)
            det_box_class_list.append(det_box_class_list1[box_i])
            det_box_geometry_list.append(det_box_geometry_list1[box_i])
            det_box_certainty_list.append(det_box_certainty_list1[box_i])

    box_pred_labels_list, completed_box_pred_labels_list \
        = segment_image_point_cloud(pc_in_box_list, frustum_angle_list, det_box_class_list, det_box_certainty_list,
                                    pc_in_box_labels_list)




    fn = np.zeros((len(REAL_CLASSES)), np.int_)
    undetected_labels = pc_labels[np.logical_not(image_box_detected_labels)]
    for type_idx in range(len(REAL_CLASSES)):
        fn += np.count_nonzero(undetected_labels == REAL_CLASSES[type_idx])


    fig = Image.fromarray(img)
    fig.show('img')
    fig.save('img.png')
    show_image_with_2d_boxes(img, det_box_geometry_list, det_box_class_list)
    show_image_with_3d_boxes(img, label_objects, calib)
    show_segmented_image(img, pts_2d_in_box_list, pc_in_box_labels_list, det_box_class_list,
                         'lidar_labels')
    if use_depth_net:
        show_segmented_image(img, pts_2d_in_box_list, pc_in_box_custom_labels_list, det_box_class_list,
                             'depth_estimation_labels')
        show_segmented_image(img, pts_2d_in_box_list, completed_box_pred_labels_list, det_box_class_list,
                             'predicted_labels')
    show_segmented_image(img, pts_2d_in_box_list, box_pred_labels_list, det_box_class_list,
                         'predicted_lidar_labels')
