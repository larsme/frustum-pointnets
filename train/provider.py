''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

# import cPickle as pickle # python 2.7
import pickle as pickle # python 3.5
import sys
import os
import numpy as np
import cv2
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR in sys.path:
   sys.path.remove(BASE_DIR)
sys.path.append(ROOT_DIR)
from train.box_util import box3d_iou
from models.model_util import g_type2class, g_class2type, g_type2onehotclass
from models.model_util import g_type_mean_size
from models.model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from kitti.kitti_object import kitti_object
from kitti.prepare_data import read_box_file, extract_pc_in_box3d, get_lidar_in_image_fov
from kitti.kitti_util import compute_box_3d

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc

# def angle2class(angle, num_class):
#     ''' Convert continuous angle to discrete class and residual.
#
#     Input:
#         angle: rad scalar, from 0-2pi (or -pi~pi), class center at
#             0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
#         num_class: int scalar, number of classes N
#     Output:
#         class_id, int, among 0,1,...,N-1
#         residual_angle: float, a number such that
#             class*(2pi/N) + residual_angle = angle
#     '''
#     angle = angle%(2*np.pi)
#     assert(angle>=0 and angle<=2*np.pi)
#     angle_per_class = 2*np.pi/float(num_class)
#     shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
#     class_id = int(shifted_angle/angle_per_class)
#     residual_angle = shifted_angle - \
#         (class_id * angle_per_class + angle_per_class/2)
#     return class_id, residual_angle
#
# def class2angle(pred_cls, residual, num_class, to_label_format=True):
#     ''' Inverse function to angle2class.
#     If to_label_format, adjust angle to the range as in labels.
#     '''
#     angle_per_class = 2*np.pi/float(num_class)
#     angle_center = pred_cls * angle_per_class
#     angle = angle_center + residual
#     if to_label_format and angle>np.pi:
#         angle = angle - 2*np.pi
#     return angle
#
# def size2class(size, type_name):
#     ''' Convert 3D bounding box size to template class and residuals.
#     todo (rqi): support multiple size clusters per type.
#
#     Input:
#         size: numpy array of shape (3,) for (l,w,h)
#         type_name: string
#     Output:
#         size_class: int scalar
#         size_residual: numpy array of shape (3,)
#     '''
#     size_class = g_type2class[type_name]
#     size_residual = size - g_type_mean_size[type_name]
#     return size_class, size_residual
#
# def class2size(pred_cls, residual):
#     ''' Inverse function to size2class. '''
#     mean_size = g_type_mean_size[g_class2type[pred_cls]]
#     return mean_size + residual


class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''
    # def __init__(self, npoints, split, classes,
    #              random_flip=False, random_shift=False, rotate_to_center=False,
    #              overwritten_data_path=None, from_rgb_detection=False, one_hot=False):
    def __init__(self, npoints, split, classes,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, box_class_one_hot=False,
                 with_intensity=True, with_color=True, vizualize_labled_images=False,
                 with_depth_confidences=False, from_guided_depth_completion=False, from_unguided_depth_completion=False,
                 from_depth_prediction=False,
                 segment_all_points=False, avoid_duplicates=False, depth_completion_augmentation=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            box_class_one_hot: bool, if True, return one hot vector
        '''
        from_depth_completion = from_guided_depth_completion or from_unguided_depth_completion
        self.use_depth_net = from_depth_prediction or from_depth_completion

        assert int(from_guided_depth_completion) + int(from_unguided_depth_completion) + int(from_depth_prediction) <= 1
        assert from_depth_completion or not with_depth_confidences
        assert not (self.use_depth_net and with_intensity)

        assert split == 'train' or not depth_completion_augmentation
        assert not with_intensity or not depth_completion_augmentation

        self.classes = classes
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.box_class_one_hot = box_class_one_hot
        self.segment_all_points = segment_all_points
        self.avoid_duplicates = avoid_duplicates
        self.depth_completion_augmentation = depth_completion_augmentation
        self.with_depth_confidences = with_depth_confidences
        self.with_color = with_color
        self.from_guided_depth_completion = from_guided_depth_completion
        self.from_unguided_depth_completion = from_unguided_depth_completion
        self.from_depth_prediction = from_depth_prediction

        if overwritten_data_path is None:
            designation = 'frustum_'
            assert len(classes) == 3
            designation += 'carpedcyc_'
            if from_unguided_depth_completion:
                designation += 'unguided_completion_'
            elif from_guided_depth_completion:
                designation += 'guided_completion_'
            elif from_depth_prediction:
                designation += 'prediction_'
            designation += split
            if from_rgb_detection:
                designation += '_rgb_detection'
            overwritten_data_path = os.path.join(ROOT_DIR, 'kitti/%s.pickle' % (designation))

        self.from_rgb_detection = from_rgb_detection

        if self.use_depth_net:
            if from_depth_completion:
                if from_guided_depth_completion:
                    bla = 0
                    # depth_net = load_net('exp_guided_nconv_cnn_l1', mode='bla', checkpoint_num=40, set_='bla')
                else: # from_unguided_depth_completion:
                    sys.path.append(os.path.join(ROOT_DIR, '../nconv'))
                    from run_nconv_cnn import load_net
                    self.depth_net = load_net('exp_unguided_depth', mode='bla', checkpoint_num=3, set_='bla')
                self.desired_image_height = 352
                self.desired_image_width = 1216
            elif from_depth_prediction:
                sys.path.append(os.path.join(ROOT_DIR, '../monodepth2'))
                from monodepth_external import load_net
                self.depth_net = load_net("mono+stereo_1024x320", use_cuda=True)

        if depth_completion_augmentation:
            self.dataset = kitti_object(os.path.join(ROOT_DIR, './../../data/kitti_object'), 'training')
            img_height_threshold = 25

            self.box_class_list = []
            self.box_geometry_list = []
            self.box_class_certainty_list = []
            self.box_image_idx_list = []
            self.frustum_angle_list = []
            if from_rgb_detection:
                rgb_det_filename = os.path.join(ROOT_DIR, 'kitti/rgb_detections/rgb_detection_%s.txt' % split)
                all_det_box_image_index_list, all_det_box_class_list, all_det_box_geometry_list, \
                    all_det_box_certainty_list = \
                    read_box_file(rgb_det_filename)
                for box_idx in range(len(all_det_box_class_list)):
                    if box_idx % 1000 == 0:
                        print('preparing box %d out of %d' %(box_idx, len(all_det_box_class_list)))
                    xmin, ymin, xmax, ymax = all_det_box_geometry_list[box_idx]
                    if all_det_box_class_list[box_idx] in classes and ymax - ymin >= img_height_threshold:
                        self.box_image_idx_list.append(all_det_box_image_index_list[box_idx])
                        self.box_class_list.append(all_det_box_class_list[box_idx])
                        self.box_geometry_list.append(all_det_box_geometry_list[box_idx])
                        self.box_class_certainty_list.append(all_det_box_certainty_list[box_idx])

                        # Get frustum angle (according to center pixel in 2D BOX)
                        calib = self.dataset.get_calibration(all_det_box_image_index_list[box_idx])  # 3 by 4 matrix
                        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                        uvdepth = np.zeros((1, 3))
                        uvdepth[0, 0:2] = box2d_center
                        uvdepth[0, 2] = 20  # some random depth
                        box2d_center_rect = calib.project_image_to_rect(uvdepth)
                        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                        box2d_center_rect[0, 0])
                        self.frustum_angle_list.append(frustum_angle)
            else:
                split_datapath = os.path.join(ROOT_DIR, 'kitti/image_sets/%s.txt' % split)
                image_idx_list = [int(line.rstrip()) for line in open(split_datapath)]
                for image_idx in image_idx_list:
                    if image_idx % 100 == 0:
                        print('preparing image %d out of %d' %(image_idx, len(image_idx_list)))
                    calib = self.dataset.get_calibration(image_idx)  # 3 by 4 matrix
                    label_objects = self.dataset.get_label_objects(image_idx)
                    for label_object in label_objects:
                        xmin, ymin, xmax, ymax = label_object.box2d
                        if label_object.type in classes and ymax - ymin >= img_height_threshold:
                            self.box_image_idx_list.append(image_idx)
                            self.box_geometry_list.append(label_object.box2d, label_object)
                            self.box_certainty_list.append(1)
                            self.box_class_list.append(label_object.type)

                            # Get frustum angle (according to center pixel in 2D BOX)
                            box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                            uvdepth = np.zeros((1, 3))
                            uvdepth[0, 0:2] = box2d_center
                            uvdepth[0, 2] = 20  # some random depth
                            box2d_center_rect = calib.project_image_to_rect(uvdepth)
                            frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                            box2d_center_rect[0, 0])
                            self.frustum_angle_list.append(frustum_angle)
        else:
            with open(overwritten_data_path, 'rb') as fp:
                # self.id_list = pickle.load(fp)
                # self.box2d_list = pickle.load(fp)
                # self.input_list = pickle.load(fp)
                # self.type_list = pickle.load(fp)
                # # frustum_angle is clockwise angle from positive x-axis
                # self.frustum_angle_list = pickle.load(fp)
                # self.prob_list = pickle.load(fp)

                # box lists
                print('box class')
                self.box_class_list = pickle.load(fp)
                print('box class certainty')
                self.box_class_certainty_list = pickle.load(fp)

                print('box point cloud')
                self.pc_in_box_list = pickle.load(fp)
                if from_depth_prediction:
                    channels = np.zeros(6, np.bool_)
                elif from_depth_completion:
                    channels = np.zeros(7, np.bool_)
                else:
                    channels = np.zeros(7, np.bool_)
                channels[:3] = 1
                i_channel = 3
                if not self.use_depth_net:
                    if with_intensity:
                        channels[i_channel] = 1
                    i_channel += 1
                elif from_depth_completion:
                    if with_depth_confidences:
                        channels[i_channel] = 1
                    i_channel += 1
                if with_color:
                    channels[i_channel:] = 1
                i_channel += 3
                if np.sum(channels) < i_channel:
                    print('removing unused channels')
                    for box_idx in range(np.size(self.pc_in_box_list, 0)):
                        self.pc_in_box_list[box_idx] = self.pc_in_box_list[box_idx][:, channels]

                print('frustum angles')
                self.frustum_angle_list = pickle.load(fp)
                print('box point cloud labels')
                self.pc_in_box_label_list = pickle.load(fp)
                print('box detection false negatives')
                self.image_boxes_fn = pickle.load(fp)

                if vizualize_labled_images:
                    print('box image ids')
                    self.box_image_idx_list = pickle.load(fp)
                    print('point cloud indices')
                    self.pc_in_box_inds_list = pickle.load(fp)
                    print('image point cloud labels')
                    self.image_pc_label_list = pickle.load(fp)

                    # load velodyne
                    # ...

                # else:
                # with open(overwritten_data_path, 'rb') as fp:
                #     self.id_list = pickle.load(fp)
                #     self.box2d_list = pickle.load(fp)
                #     self.box3d_list = pickle.load(fp)
                #     self.input_list = pickle.load(fp)
                #     self.label_list = pickle.load(fp)
                #     self.type_list = pickle.load(fp)
                #     self.heading_list = pickle.load(fp)
                #     self.size_list = pickle.load(fp)
                #     # frustum_angle is clockwise angle from positive x-axis
                #     self.frustum_angle_list = pickle.load(fp)

    def __len__(self):
        return len(self.box_class_list)

    def __getitem__(self, opts):
        ''' Get box_index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        box_index, left_to_sample = opts

        # Compute one hot vector
        if self.box_class_one_hot:
            box_class = self.box_class_list[box_index]
            assert (box_class in self.classes)
            box_class_one_hot_vec = np.zeros((len(self.classes)), np.bool_)
            box_class_one_hot_vec[g_type2onehotclass[box_class]] = 1

        if self.depth_completion_augmentation:
            assert left_to_sample == []

            image_index = self.box_image_idx_list[box_index]
            calib = self.dataset.get_calibration(image_index)  # 3 by 4 matrix
            img = self.dataset.get_image(image_index)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width, img_channel = img.shape
            res_dir = os.path.join(ROOT_DIR, '../../data/completed_depth')

            dense_depths = []
            confidences = []
            if self.depth_completion_augmentation:
                (dense_depths, confidences) = np.load(os.path.join(res_dir, str(image_index) + '.npy'))

            # 2D BOX: Get pts rect backprojected
            xmin, ymin, xmax, ymax = self.box_geometry_list[box_index]

            int_x_min = int(max(np.floor(xmin), 0))
            int_x_max = int(min(np.ceil(xmax), img_width - 1))
            box_x_width = int_x_max - int_x_min + 1

            int_y_min = int(max(np.floor(ymin), 0))
            int_y_max = int(min(np.ceil(ymax), img_height - 1))
            box_y_width = int_y_max - int_y_min + 1

            box_sub_pixels_row, box_sub_pixels_col = np.indices((box_y_width, box_x_width))
            box_sub_pixels_row = np.reshape(box_sub_pixels_row, -1)
            box_sub_pixels_col = np.reshape(box_sub_pixels_col, -1)

            choice, idontcare, left_to_sample = self.get_choice(left_to_sample, box_sub_pixels_row.shape[0])

            box_sub_pixels_row = box_sub_pixels_row[choice]
            box_sub_pixels_col = box_sub_pixels_col[choice]
            pixels_in_box_row = box_sub_pixels_row + int_y_min
            pixels_in_box_col = box_sub_pixels_col + int_x_min

            depths_in_box = dense_depths[pixels_in_box_row, pixels_in_box_col]
            new_pc_img_in_box = np.concatenate((np.ndarray.astype(np.expand_dims(pixels_in_box_col, 1), np.float),
                                                np.ndarray.astype(np.expand_dims(pixels_in_box_row, 1), np.float),
                                                np.expand_dims(depths_in_box, 1)), axis=1)
            new_pc_rect_in_box = calib.project_image_to_rect(new_pc_img_in_box)

            pc_in_box_gt_labels = np.zeros((np.size(new_pc_rect_in_box, 0)), np.int) - 1

            for label_object in self.dataset.get_label_objects(image_index):
                _, box3d_pts_3d = compute_box_3d(label_object, calib.P)
                _, instance_pc_indexes = extract_pc_in_box3d(new_pc_rect_in_box, box3d_pts_3d)
                overlapping_3d_boxes = np.nonzero(pc_in_box_gt_labels[instance_pc_indexes])[0]
                pc_in_box_gt_labels[instance_pc_indexes] = 1
                (pc_in_box_gt_labels[instance_pc_indexes])[overlapping_3d_boxes] = -1

            input_pc = new_pc_rect_in_box
            if self.with_depth_confidences:
                confidences_in_box = np.expand_dims(confidences[pixels_in_box_row, pixels_in_box_col], 1)
                input_pc = np.concatenate((input_pc, confidences_in_box), axis=1)
            if self.with_color:
                pc_in_box_colors = img[pixels_in_box_row, pixels_in_box_col, :]
                input_pc = np.concatenate((input_pc, pc_in_box_colors), axis=1)
        else:
            input_pc = self.pc_in_box_list[box_index]
            choice, idontcare, left_to_sample = self.get_choice(left_to_sample, input_pc.shape[0])
            input_pc = input_pc[choice, :]

            # if self.from_rgb_detection:
            #     if self.box_class_one_hot:
            #         return input_pc, rot_angle, self.box_class_certainty_list[box_index], box_class_one_hot_vec
            #     else:
            #         return input_pc, rot_angle, self.box_class_certainty_list[box_index]

            # ------------------------------ LABELS ----------------------------
            pc_in_box_gt_labels = np.squeeze(self.pc_in_box_label_list[box_index][choice])
            pc_in_box_gt_labels[idontcare] = -1

        rot_angle = self.get_center_view_rot_angle(box_index)
        if self.rotate_to_center:
            input_pc = rotate_pc_along_y(input_pc, rot_angle)

        # # Get center point of 3D box
        # if self.rotate_to_center:
        #     box3d_center = self.get_center_view_box3d_center(box_index)
        # else:
        #     box3d_center = self.get_box3d_center(box_index)
        #
        # # Heading
        # if self.rotate_to_center:
        #     heading_angle = self.heading_list[box_index] - rot_angle
        # else:
        #     heading_angle = self.heading_list[box_index]
        #
        # # Size
        # size_class, size_residual = size2class(self.size_list[box_index],
        #     self.type_list[box_index])

        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random()>0.5: # 50% chance flipping
                input_pc[:, 0] *= -1
                # box3d_center[0] *= -1
                # heading_angle = np.pi - heading_angle
        if self.random_shift:
            # dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
            dist = np.sqrt(np.sum(np.mean(self.pc_in_box_list[:, 0])**2 + np.mean(self.pc_in_box_list[:, 1])**2))
            shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
            input_pc[:, 2] += shift
            # box3d_center[2] += shift

        # angle_class, angle_residual = angle2class(heading_angle,
        #     NUM_HEADING_BIN)

        # if self.box_class_one_hot:
        #     return input_pc, pc_in_box_gt_labels, box3d_center, angle_class, angle_residual,\
        #         size_class, size_residual, rot_angle, box_class_one_hot_vec
        # else:
        #     return input_pc, pc_in_box_gt_labels, box3d_center, angle_class, angle_residual,\
        #         size_class, size_residual, rot_angle

        if self.from_rgb_detection:
            if self.box_class_one_hot:
                return [input_pc, pc_in_box_gt_labels, rot_angle, self.box_class_certainty_list[box_index],
                        left_to_sample, box_class_one_hot_vec]
            else:
                return [input_pc, pc_in_box_gt_labels, rot_angle, self.box_class_certainty_list[box_index],
                        left_to_sample]

    def get_choice(self, left_to_sample, nLidarPoints):
        # Resample
        if self.segment_all_points:
            if len(left_to_sample) <= self.npoints:
                choice = np.zeros(self.npoints, np.int_)
                choice[0:len(left_to_sample)] = left_to_sample
                if self.avoid_duplicates and nLidarPoints >= self.npoints:
                    sampled = np.delete(range(nLidarPoints), left_to_sample)
                    choice[len(left_to_sample):self.npoints] = np.random.choice(sampled,
                                                                                self.npoints - len(left_to_sample),
                                                                                replace=False)
                else:
                    choice[len(left_to_sample):self.npoints] = np.random.choice(nLidarPoints,
                                                                                self.npoints - len(left_to_sample),
                                                                                replace=True)
                idontcare = np.zeros(self.npoints, np.bool_)
                idontcare[len(left_to_sample):self.npoints] = True
                permut = np.random.permutation(range(self.npoints))
                choice = choice[permut]
                idontcare = idontcare[permut]
                left_to_sample = []
            else:
                choice = np.random.permutation(left_to_sample)
                left_to_sample = choice[self.npoints:]
                choice = choice[:self.npoints]
                idontcare = np.zeros(self.npoints, np.bool_)
        else:
            if self.avoid_duplicates and nLidarPoints >= self.npoints:
                choice = np.random.choice(nLidarPoints, self.npoints, replace=False)
            else:
                choice = np.random.choice(nLidarPoints, self.npoints, replace=True)
            idontcare = np.zeros(self.npoints, np.bool_)
        return choice, idontcare, left_to_sample

    def show_points_per_box_statistics(self):
        import matplotlib.pyplot as plt

        points_per_box = []
        points_per_box.append([])
        points_per_box.append([])
        points_per_box.append([])
        points_per_box_ges = []
        for box_idx in range(np.size(self.pc_in_box_list, 0)):
            points_per_box[g_type2onehotclass[self.box_class_list[box_idx]]].append(
                np.size(self.pc_in_box_list[box_idx], 0))
            points_per_box_ges.append(
                np.size(self.pc_in_box_list[box_idx], 0))

        print(self.classes[0])
        print(len(points_per_box[0]))
        print(np.mean(points_per_box[0]))
        print(np.var(points_per_box[0]))
        plt.hist(points_per_box[0], bins='auto')
        plt.title('Points per box of type ' + self.classes[0])
        plt.savefig('Points per box of type ' + self.classes[0])
        plt.show()
        print()

        raw_input()

        print(self.classes[1])
        print(len(points_per_box[1]))
        print(np.mean(points_per_box[1]))
        print(np.var(points_per_box[1]))
        plt.hist(points_per_box[1], bins='auto')
        plt.title('Points per box of type ' + self.classes[1])
        plt.savefig('Points per box of type ' + self.classes[1])
        plt.show()
        print()

        raw_input()

        print(self.classes[2])
        print(len(points_per_box[2]))
        print(np.mean(points_per_box[2]))
        print(np.var(points_per_box[2]))
        plt.hist(points_per_box[2], bins='auto')
        plt.title('Points per box of type ' + self.classes[2])
        plt.savefig('Points per box of type ' + self.classes[2])
        plt.show()
        print()

        raw_input()

        print('Ges')
        plt.hist(points_per_box_ges, bins='auto')
        plt.title('Points per box of any type')
        plt.savefig('Points per box of any type')
        plt.show()
        print(len(points_per_box_ges))
        print(np.mean(points_per_box_ges))
        print(np.var(points_per_box_ges))
        print()

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it is shifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]

    # def get_box3d_center(self, index):
    #     ''' Get the center (XYZ) of 3D bounding box. '''
    #     box3d_center = (self.box3d_list[index][0,:] + \
    #         self.box3d_list[index][6,:])/2.0
    #     return box3d_center
    #
    # def get_center_view_box3d_center(self, index):
    #     ''' Frustum rotation of 3D bounding box center. '''
    #     box3d_center = (self.box3d_list[index][0,:] + \
    #         self.box3d_list[index][6,:])/2.0
    #     return rotate_pc_along_y(np.expand_dims(box3d_center,0), \
    #         self.get_center_view_rot_angle(index)).squeeze()
    #
    # def get_center_view_box3d(self, index):
    #     ''' Frustum rotation of 3D bounding box corners. '''
    #     box3d = self.box3d_list[index]
    #     box3d_center_view = np.copy(box3d)
    #     return rotate_pc_along_y(box3d_center_view, \
    #         self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.pc_in_box_list[index])
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle(index))


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

# def get_3d_box(box_size, heading_angle, center):
#     ''' Calculate 3D bounding box corners from its parameterization.
#
#     Input:
#         box_size: tuple of (l,w,h)
#         heading_angle: rad scalar, clockwise from pos x axis
#         center: tuple of (x,y,z)
#     Output:
#         corners_3d: numpy array of shape (8,3) for 3D box cornders
#     '''
#     def roty(t):
#         c = np.cos(t)
#         s = np.sin(t)
#         return np.array([[c,  0,  s],
#                          [0,  1,  0],
#                          [-s, 0,  c]])
#
#     R = roty(heading_angle)
#     l,w,h = box_size
#     x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
#     y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
#     z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
#     corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
#     corners_3d[0,:] = corners_3d[0,:] + center[0];
#     corners_3d[1,:] = corners_3d[1,:] + center[1];
#     corners_3d[2,:] = corners_3d[2,:] + center[2];
#     corners_3d = np.transpose(corners_3d)
#     return corners_3d
#
# def compute_box3d_iou(center_pred,
#                       heading_logits, heading_residuals,
#                       size_logits, size_residuals,
#                       center_label,
#                       heading_class_label, heading_residual_label,
#                       size_class_label, size_residual_label):
#     ''' Compute 3D bounding box IoU from network output and labels.
#     All inputs are numpy arrays.
#
#     Inputs:
#         center_pred: (B,3)
#         heading_logits: (B,NUM_HEADING_BIN)
#         heading_residuals: (B,NUM_HEADING_BIN)
#         size_logits: (B,NUM_SIZE_CLUSTER)
#         size_residuals: (B,NUM_SIZE_CLUSTER,3)
#         center_label: (B,3)
#         heading_class_label: (B,)
#         heading_residual_label: (B,)
#         size_class_label: (B,)
#         size_residual_label: (B,3)
#     Output:
#         iou2ds: (B,) birdeye view oriented 2d box ious
#         iou3ds: (B,) 3d box ious
#     '''
#     batch_size = heading_logits.shape[0]
#     heading_class = np.argmax(heading_logits, 1) # B
#     heading_residual = np.array([heading_residuals[i,heading_class[i]] \
#         for i in range(batch_size)]) # B,
#     size_class = np.argmax(size_logits, 1) # B
#     size_residual = np.vstack([size_residuals[i,size_class[i],:] \
#         for i in range(batch_size)])
#
#     iou2d_list = []
#     iou3d_list = []
#     for i in range(batch_size):
#         heading_angle = class2angle(heading_class[i],
#             heading_residual[i], NUM_HEADING_BIN)
#         box_size = class2size(size_class[i], size_residual[i])
#         corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])
#
#         heading_angle_label = class2angle(heading_class_label[i],
#             heading_residual_label[i], NUM_HEADING_BIN)
#         box_size_label = class2size(size_class_label[i], size_residual_label[i])
#         corners_3d_label = get_3d_box(box_size_label,
#             heading_angle_label, center_label[i])
#
#         iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
#         iou3d_list.append(iou_3d)
#         iou2d_list.append(iou_2d)
#     return np.array(iou2d_list, dtype=np.float32), \
#         np.array(iou3d_list, dtype=np.float32)
#
#
# def from_prediction_to_label_format(center, angle_class, angle_res,\
#                                     size_class, size_res, rot_angle):
#     ''' Convert predicted box parameters to label format. '''
#     l,w,h = class2size(size_class, size_res)
#     ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
#     tx,ty,tz = rotate_pc_along_y(np.expand_dims(center,0),-rot_angle).squeeze()
#     ty += h/2.0
#     return h,w,l,tx,ty,tz,ry



if __name__=='__main__':
    import mayavi.mlab as mlab 
    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_lidar, draw_gt_boxes3d
    median_list = []
    dataset = FrustumDataset(1024, split='val',
        rotate_to_center=True, random_flip=True, random_shift=True)
    for i in range(len(dataset)):
        data = dataset[i]
        print(('Center: ', data[2], \
            'angle_class: ', data[3], 'angle_res:', data[4], \
            'size_class: ', data[5], 'size_residual:', data[6], \
            'real_size:', g_type_mean_size[g_class2type[data[5]]]+data[6]))
        print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        median_list.append(np.median(data[0][:, 0]))
        print((data[2], dataset.box3d_list[i], median_list[-1]))
        # box3d_from_label = get_3d_box(class2size(data[5],data[6]), class2angle(data[3], data[4],12), data[2])

        ps = data[0]
        seg = data[1]
        fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:,0], ps[:,1], ps[:,2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2, figure=fig)
        # draw_gt_boxes3d([box3d_from_label], fig, color=(1,0,0))
        mlab.orientation_axes()
        mlab.savefig('draw_line.jpg', figure=fig)
        raw_input()
    print(np.mean(np.abs(median_list)))
