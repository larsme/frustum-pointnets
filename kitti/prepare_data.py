''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import kitti.kitti_util as utils
import pickle as pickle # python 3.5
# import cPickle as pickle # python 2.7
from kitti.kitti_object import *
import argparse


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]] 
    box2d_corners[1,:] = [box2d[2],box2d[1]] 
    box2d_corners[2,:] = [box2d[2],box2d[3]] 
    box2d_corners[3,:] = [box2d[0],box2d[3]] 
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds
     
def demo():
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = kitti_object('./../../data/kitti_object')
    data_idx = 0

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    pc_velo = dataset.get_lidar(data_idx)[:,0:3]
    calib = dataset.get_calibration(data_idx)

    ## Draw lidar in rect camera coord
    #print(' -------- LiDAR points in rect camera coordination --------')
    #pc_rect = calib.project_velo_to_rect(pc_velo)
    #fig = draw_lidar_simple(pc_rect)
    #raw_input()

    # Draw 2d and 3d boxes on image
    print(' -------- 2D/3D bounding boxes in images --------')
    show_image_with_boxes(img, objects, calib)
    raw_input()

    # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    #show_lidar_with_boxes(pc_velo, objects, calib)
    #raw_input()
    show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
    raw_input()

    # Visualize LiDAR points on images
    print(' -------- LiDAR points projected to image plane --------')
    show_lidar_on_image(pc_velo, img, calib, img_width, img_height) 
    raw_input()
    
    # Show LiDAR points that are in the 3d box
    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P) 
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.savefig('draw_line.jpg', figure=fig)
    mlab.show(1)
    raw_input()
    
    # UVDepth Image and its backprojection to point clouds
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:,0:2] = imgfov_pts_2d
    cameraUVDepth[:,2] = imgfov_pc_rect[:,2]

    # Show that the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print(imgfov_pc_velo[0:20])
    print(backprojected_pc_velo[0:20])

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into 2d box
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    xmin,ymin,xmax,ymax = \
        objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = \
        get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
    print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.savefig('draw_line.jpg', figure=fig)
    mlab.show(1)
    raw_input()

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])
 
# def extract_frustum_data(idx_filename, split, output_filename, viz=False,
#                        perturb_box2d=False, augment_x=1, type_whitelist=['Car']):
#     ''' Extract point clouds and corresponding annotations in frustums
#         defined generated from 2D bounding boxes
#         Lidar points and 3d boxes are in *rect camera* coord system
#         (as that in 3d box label files)
#
#     Input:
#         idx_filename: string, each line of the file is a sample ID
#         split: string, either training or testing
#         output_filename: string, the name for output .pickle file
#         viz: bool, whether to visualize extracted data
#         perturb_box2d: bool, whether to perturb the box2d
#             (used for data augmentation in train set)
#         augment_x: scalar, how many augmentations to have for each 2D box.
#         type_whitelist: a list of strings, object types we are interested in.
#     Output:
#         None (will write a .pickle file to the disk)
#     '''
#     dataset = kitti_object(os.path.join(ROOT_DIR,'./../../data/kitti_object'), split)
#     data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
#
#     id_list = [] # int number
#     box2d_list = [] # [xmin,ymin,xmax,ymax]
#     box3d_list = [] # (8,3) array in rect camera coord
#     input_list = [] # channel number = 4, xyz,intensity in rect camera coord
#     label_list = [] # 1 for roi object, 0 for clutter
#     type_list = [] # string e.g. Car
#     heading_list = [] # ry (along y-axis in rect camera coord) radius of
#     # (cont.) clockwise angle from positive x axis in velo coord.
#     box3d_size_list = [] # array of l,w,h
#     frustum_angle_list = [] # angle of 2d box center from pos x-axis
#
#     pos_cnt = 0
#     all_cnt = 0
#     for data_idx in data_idx_list:
#         print('------------- ', data_idx)
#         calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
#         objects = dataset.get_label_objects(data_idx)
#         pc_velo = dataset.get_lidar(data_idx)
#         pc_rect = np.zeros_like(pc_velo)
#         pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
#         pc_rect[:,3] = pc_velo[:,3]
#         img = dataset.get_image(data_idx)
#         img_height, img_width, img_channel = img.shape
#         _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
#             calib, 0, 0, img_width, img_height, True)
#
#         for obj_idx in range(len(objects)):
#             if objects[obj_idx].type not in type_whitelist :continue
#
#             # 2D BOX: Get pts rect backprojected
#             box2d = objects[obj_idx].box2d
#             for _ in range(augment_x):
#                 # Augment data by box2d perturbation
#                 if perturb_box2d:
#                     xmin,ymin,xmax,ymax = random_shift_box2d(box2d)
#                     print(box2d)
#                     print(xmin,ymin,xmax,ymax)
#                 else:
#                     xmin,ymin,xmax,ymax = box2d
#                 box_fov_inds = (pc_image_coord[:,0]<xmax) & \
#                     (pc_image_coord[:,0]>=xmin) & \
#                     (pc_image_coord[:,1]<ymax) & \
#                     (pc_image_coord[:,1]>=ymin)
#                 box_fov_inds = box_fov_inds & img_fov_inds
#                 pc_in_box_fov = pc_rect[box_fov_inds,:]
#                 # Get frustum angle (according to center pixel in 2D BOX)
#                 box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
#                 uvdepth = np.zeros((1,3))
#                 uvdepth[0,0:2] = box2d_center
#                 uvdepth[0,2] = 20 # some random depth
#                 box2d_center_rect = calib.project_image_to_rect(uvdepth)
#                 frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
#                     box2d_center_rect[0,0])
#                 # 3D BOX: Get pts velo in 3d box
#                 obj = objects[obj_idx]
#                 box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
#                 _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
#                 label = np.zeros((pc_in_box_fov.shape[0]))
#                 label[inds] = 1
#                 # Get 3D BOX heading
#                 heading_angle = obj.ry
#                 # Get 3D BOX size
#                 box3d_size = np.array([obj.l, obj.w, obj.h])
#
#                 # Reject too far away object or object without points
#                 if ymax-ymin<25 or np.sum(label)==0:
#                     continue
#
#                 id_list.append(data_idx)
#                 box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
#                 box3d_list.append(box3d_pts_3d)
#                 input_list.append(pc_in_box_fov)
#                 label_list.append(label)
#                 type_list.append(objects[obj_idx].type)
#                 heading_list.append(heading_angle)
#                 box3d_size_list.append(box3d_size)
#                 frustum_angle_list.append(frustum_angle)
#
#                 # collect statistics
#                 pos_cnt += np.sum(label)
#                 all_cnt += pc_in_box_fov.shape[0]
#
#     print('Average pos ratio: %f' % (pos_cnt/float(all_cnt)))
#     print('Average npoints: %f' % (float(all_cnt)/len(id_list)))
#
#     with open(output_filename,'wb') as fp:
#         pickle.dump(id_list, fp)
#         pickle.dump(box2d_list,fp)
#         pickle.dump(box3d_list,fp)
#         pickle.dump(input_list, fp)
#         pickle.dump(label_list, fp)
#         pickle.dump(type_list, fp)
#         pickle.dump(heading_list, fp)
#         pickle.dump(box3d_size_list, fp)
#         pickle.dump(frustum_angle_list, fp)
#
#     if viz:
#         import mayavi.mlab as mlab
#         for i in range(10):
#             p1 = input_list[i]
#             seg = label_list[i]
#             fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
#                 fgcolor=None, engine=None, size=(500, 500))
#             mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point',
#                 colormap='gnuplot', scale_factor=1, figure=fig)
#             fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
#                 fgcolor=None, engine=None, size=(500, 500))
#             mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
#                 colormap='gnuplot', scale_factor=1, figure=fig)
#             raw_input()

# def get_box3d_dim_statistics(idx_filename):
#     ''' Collect and dump 3D bounding box statistics '''
#     dataset = kitti_object(os.path.join(ROOT_DIR,'./../../data/kitti_object'))
#     dimension_list = []
#     type_list = []
#     ry_list = []
#     data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
#     for data_idx in data_idx_list:
#         print('------------- ', data_idx)
#         calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
#         objects = dataset.get_label_objects(data_idx)
#         for obj_idx in range(len(objects)):
#             obj = objects[obj_idx]
#             if obj.type=='DontCare':continue
#             dimension_list.append(np.array([obj.l,obj.w,obj.h]))
#             type_list.append(obj.type)
#             ry_list.append(obj.ry)
#
#     with open('box3d_dimensions.pickle','wb') as fp:
#         pickle.dump(type_list, fp)
#         pickle.dump(dimension_list, fp)
#         pickle.dump(ry_list, fp)

def read_box_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3,7)]))
    return id_list, type_list, box2d_list, prob_list


# def extract_frustum_data_rgb_detection(det_filename, split, output_filename,
#                                        viz=False,
#                                        type_whitelist=['Car'],
#                                        img_height_threshold=25,
#                                        lidar_point_threshold=5):
#     ''' Extract point clouds in frustums extruded from 2D detection boxes.
#         Update: Lidar points and 3d boxes are in *rect camera* coord system
#             (as that in 3d box label files)
#
#     Input:
#         det_filename: string, each line is
#             img_path typeid confidence xmin ymin xmax ymax
#         split: string, either trianing or testing
#         output_filename: string, the name for output .pickle file
#         type_whitelist: a list of strings, object types we are interested in.
#         img_height_threshold: int, neglect image with height lower than that.
#         lidar_point_threshold: int, neglect frustum with too few points.
#     Output:
#         None (will write a .pickle file to the disk)
#     '''
#     dataset = kitti_object(os.path.join(ROOT_DIR, './../../data/kitti_object'), split)
#     det_id_list, det_type_list, det_box2d_list, det_prob_list = \
#         read_det_file(det_filename)
#     cache_id = -1
#     cache = None
#
#     id_list = []
#     type_list = []
#     box2d_list = []
#     prob_list = []
#     input_list = []  # channel number = 4, xyz,intensity in rect camera coord
#     frustum_angle_list = [] # angle of 2d box center from pos x-axis
#
#     for det_idx in range(len(det_id_list)):
#         data_idx = det_id_list[det_idx]
#         print('det idx: %d/%d, data idx: %d' % \
#               (det_idx, len(det_id_list), data_idx))
#         if cache_id != data_idx:
#             calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
#             pc_velo = dataset.get_lidar(data_idx)
#             pc_rect = np.zeros_like(pc_velo)
#             pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
#             pc_rect[:, 3] = pc_velo[:, 3]
#             img = dataset.get_image(data_idx)
#             img_height, img_width, img_channel = img.shape
#             _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov( \
#                 pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
#             cache = [calib, pc_rect, pc_image_coord, img_fov_inds]
#             cache_id = data_idx
#         else:
#             calib, pc_rect, pc_image_coord, img_fov_inds = cache
#
#         if det_type_list[det_idx] not in type_whitelist: continue
#
#         # 2D BOX: Get pts rect backprojected
#         xmin, ymin, xmax, ymax = det_box2d_list[det_idx]
#         box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
#                        (pc_image_coord[:, 0] >= xmin) & \
#                        (pc_image_coord[:, 1] < ymax) & \
#                        (pc_image_coord[:, 1] >= ymin)
#         box_fov_inds = box_fov_inds & img_fov_inds
#         pc_in_box_fov = pc_rect[box_fov_inds, :]
#         # Get frustum angle (according to center pixel in 2D BOX)
#         box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
#         uvdepth = np.zeros((1, 3))
#         uvdepth[0, 0:2] = box2d_center
#         uvdepth[0, 2] = 20  # some random depth
#         box2d_center_rect = calib.project_image_to_rect(uvdepth)
#         frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
#                                         box2d_center_rect[0, 0])
#
#         # Pass objects that are too small
#         if ymax - ymin < img_height_threshold or \
#                 len(pc_in_box_fov) < lidar_point_threshold:
#             continue
#
#         id_list.append(data_idx)
#         type_list.append(det_type_list[det_idx])
#         box2d_list.append(det_box2d_list[det_idx])
#         prob_list.append(det_prob_list[det_idx])
#         input_list.append(pc_in_box_fov)
#         frustum_angle_list.append(frustum_angle)
#
#     with open(output_filename, 'wb') as fp:
#         pickle.dump(id_list, fp)
#         pickle.dump(box2d_list, fp)
#         pickle.dump(input_list, fp)
#         pickle.dump(type_list, fp)
#         pickle.dump(frustum_angle_list, fp)
#         pickle.dump(prob_list, fp)
#
#     if viz:
#         import mayavi.mlab as mlab
#         for i in range(10):
#             p1 = input_list[i]
#             fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
#                               fgcolor=None, engine=None, size=(500, 500))
#             mlab.points3d(p1[:, 0], p1[:, 1], p1[:, 2], p1[:, 1], mode='point',
#                           colormap='gnuplot', scale_factor=1, figure=fig)
#             fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
#                               fgcolor=None, engine=None, size=(500, 500))
#             mlab.points3d(p1[:, 2], -p1[:, 0], -p1[:, 1], seg, mode='point',
#                           colormap='gnuplot', scale_factor=1, figure=fig)
#             raw_input()

def show_points_per_box_statistics(split_file_datapath,
                     split,
                     train_split,
                     type_whitelist=['Car'],
                     from_rgb_detection=True,
                     rgb_det_filename="",
                     img_height_threshold=25):
    import tikzplotlib
    import matplotlib.pyplot as plt

    image_idx_list = [int(line.rstrip()) for line in open(split_file_datapath)]
    dataset = kitti_object(os.path.join(ROOT_DIR, './../../data/kitti_object'), split)

    points_per_box = []
    ges_points_per_box = []
    for i in range(len(type_whitelist)):
        points_per_box.append([])

    if not from_rgb_detection:
        for image_idx in image_idx_list:
            label_objects = dataset.get_label_objects(image_idx)
            for label_object in label_objects:
                for i in range(len(type_whitelist)):
                    if type_whitelist[i] == label_object.type:
                        xmin, ymin, xmax, ymax = label_object.box2d
                        if ymax - ymin >= img_height_threshold:
                            points_per_box[i].append((xmax-xmin)*(ymax-ymin))
                            ges_points_per_box.append((xmax - xmin) * (ymax - ymin))
                        break

    if from_rgb_detection:
        _, det_box_class_list, det_box_geometry_list, _ = \
            read_box_file(rgb_det_filename)

        for box_idx in range(len(det_box_class_list)):
            for i in range(len(type_whitelist)):
                if type_whitelist[i] == det_box_class_list[box_idx]:
                    xmin, ymin, xmax, ymax = det_box_geometry_list[box_idx]
                    if ymax - ymin >= img_height_threshold:
                        points_per_box[i].append((xmax - xmin) * (ymax - ymin))
                        ges_points_per_box.append((xmax - xmin) * (ymax - ymin))
                    break


    for i in range(len(type_whitelist)):
        print(type_whitelist[i])
        print(len(points_per_box[i]))
        print(np.mean(points_per_box[i]))
        print(np.var(points_per_box[i]))
        plt.hist(points_per_box[i], bins='auto')
        plt.title('Pixels per box of type ' + type_whitelist[i])
        plt.savefig(train_split+' - '+'Pixels per box of type ' + type_whitelist[i])
        tikzplotlib.save(train_split+' - '+'Pixels per box of type ' + type_whitelist[i]+".tikz")
        plt.show()
        print()

    print('Ges')
    plt.hist(ges_points_per_box, bins='auto')
    plt.title('Pixels per box of any type')
    plt.savefig(train_split+' - '+'Pixels per box of any type')
    tikzplotlib.save(train_split+' - '+'Pixels per box of any type.tikz')
    plt.show()
    print(len(ges_points_per_box))
    print(np.mean(ges_points_per_box))
    print(np.var(ges_points_per_box))
    print()


def extract_frustum_data(split_file_datapath,
                         split, output_filename,
                         viz=False, perturb_box2d=False, augment_x=1,
                         type_whitelist=['Car'],
                         from_rgb_detection=True,
                         rgb_det_filename="",
                         img_height_threshold=25,
                         lidar_point_threshold=5,
                         from_unguided_depth_completion=False,
                         from_guided_depth_completion=False,
                         from_depth_prediction=False,
                         fill_n_points=-1):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        split_file_datapath: string, each line of the file is a image sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augment_x: scalar, how many augmentations to have for each 2D box (no augmentation => 1).
        rgb_det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''

    assert augment_x > 0
    if not perturb_box2d:
        augment_x = 1

    from_depth_completion = from_unguided_depth_completion or from_guided_depth_completion
    use_depth_net = from_depth_completion or from_depth_prediction
    assert int(from_guided_depth_completion) + int(from_unguided_depth_completion) + int(from_depth_prediction) <= 1
    assert use_depth_net or fill_n_points == -1

    if from_depth_completion:
        if from_guided_depth_completion:
            bla = 0
            # depth_net = load_net('exp_guided_nconv_cnn_l1', mode='bla', checkpoint_num=40, set_='bla')
        else: # from_unguided_depth_completion:
            sys.path.append(os.path.join(ROOT_DIR, '../nconv'))
            from run_nconv_cnn import load_net
            depth_net = load_net('exp_unguided_depth', mode='bla', checkpoint_num=3, set_='bla')
        desired_image_height = 352
        desired_image_width = 1216
    elif from_depth_prediction:
        sys.path.append(os.path.join(ROOT_DIR, '../monodepth2'))
        from monodepth_external import load_net
        depth_net = load_net("mono+stereo_1024x320", use_cuda=True)

    image_idx_list = [int(line.rstrip()) for line in open(split_file_datapath)]
    dataset = kitti_object(os.path.join(ROOT_DIR, './../../data/kitti_object'), split)

    # image labels
    image_pc_label_list = []
    image_box_detected_label_list = []

    if not from_rgb_detection:
        det_box_image_index_list = []
        det_box_class_list = []
        det_box_geometry_list = []
        det_box_certainty_list = []

    o_filler = np.zeros(0, np.object)
    b_filler = np.zeros((augment_x, 0), np.bool_)
    for image_idx in range(dataset.num_samples):
        image_pc_label_list.append(o_filler)
        image_box_detected_label_list.append(b_filler)

    for image_idx in image_idx_list:
        print('image idx: %d/%d' % \
              ( image_idx, dataset.num_samples))
        calib = dataset.get_calibration(image_idx)  # 3 by 4 matrix

        pc_velo = dataset.get_lidar(image_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]

        img = dataset.get_image(image_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        _, _, img_fov_inds, _ = get_lidar_in_image_fov( \
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
        pc_rect = pc_rect[img_fov_inds, :]

        label_objects = dataset.get_label_objects(image_idx)
        pc_labels = np.zeros((np.size(pc_rect, 0)), np.object)
        for label_object in label_objects:
            _, box3d_pts_3d = utils.compute_box_3d(label_object, calib.P)
            _, instance_pc_indexes = extract_pc_in_box3d(pc_rect, box3d_pts_3d)
            overlapping_3d_boxes = np.nonzero(pc_labels[instance_pc_indexes])[0]
            pc_labels[instance_pc_indexes] = label_object.type
            (pc_labels[instance_pc_indexes])[overlapping_3d_boxes] = 'DontCare'
            xmin, ymin, xmax, ymax = label_object.box2d
            if not from_rgb_detection and label_object.type in type_whitelist and ymax - ymin >= img_height_threshold:
                det_box_geometry_list.append(label_object.box2d)
                det_box_certainty_list.append(1)
                det_box_class_list.append(label_object.type)
                det_box_image_index_list.append(image_idx)

        image_pc_label_list[image_idx] = pc_labels
        image_box_detected_label_list[image_idx] = np.zeros((pc_labels.shape[0], augment_x), np.bool_)

    if from_rgb_detection:
        all_det_box_image_index_list, all_det_box_class_list, all_det_box_geometry_list, \
        all_det_box_certainty_list = \
            read_box_file(rgb_det_filename)
        det_box_image_index_list = []
        det_box_class_list = []
        det_box_geometry_list = []
        det_box_certainty_list = []
        for box_idx in range(len(all_det_box_class_list)):
            xmin, ymin, xmax, ymax = all_det_box_geometry_list[box_idx]
            if all_det_box_class_list[box_idx] in type_whitelist and ymax - ymin >= img_height_threshold:
                det_box_image_index_list.append(all_det_box_image_index_list[box_idx])
                det_box_class_list.append(all_det_box_class_list[box_idx])
                det_box_geometry_list.append(all_det_box_geometry_list[box_idx])
                det_box_certainty_list.append(all_det_box_certainty_list[box_idx])

    cache_id = -1
    cache = None

    box_class_list = []
    box_certainty_list = []
    pc_in_box_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis
    pc_in_box_label_list = []

    box_image_id_list = []
    pc_in_box_inds_list = []

    for box_idx in range(len(det_box_image_index_list)):
        image_idx = det_box_image_index_list[box_idx]
        print('box idx: %d/%d, image idx: %d' % \
              (box_idx, len(det_box_image_index_list), image_idx))
        if cache_id != image_idx:
            calib = dataset.get_calibration(image_idx)  # 3 by 4 matrix

            pc_velo = dataset.get_lidar(image_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
            pc_rect[:, 3] = pc_velo[:, 3]

            img = dataset.get_image(image_idx)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width, img_channel = img.shape

            _, pts_image_2d, img_fov_inds, pc_image_depths = get_lidar_in_image_fov( \
                pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
            pc_rect = pc_rect[img_fov_inds, :]
            pts_image_2d = np.ndarray.astype(np.round(pts_image_2d[img_fov_inds, :]), int)
            pts_image_2d[pts_image_2d < 0] = 0
            pts_image_2d[pts_image_2d[:, 0] >= img_width, 0] = img_width-1
            pts_image_2d[pts_image_2d[:, 1] >= img_height, 1] = img_height-1
            pc_labels = image_pc_label_list[image_idx]
            pc_image_depths = pc_image_depths[img_fov_inds]

            dense_depths = []
            confidences = []
            if from_unguided_depth_completion:
                lidarmap = dataset.generate_depth_map(image_idx, 2, desired_image_width, desired_image_height)
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
                dense_depths, confidences = depth_net.return_one_prediction(lidarmap*255, rgb, img_width, img_height)
            elif from_guided_depth_completion:
                res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/completed_depth')
                (dense_depths, confidences) = np.load(os.path.join(res_dir, str(image_idx)+'.npy'))

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
            elif from_depth_prediction:
                dense_depths = depth_net.return_one_prediction(img, post_process=False)

            cache = [calib, pc_rect, pts_image_2d, img_height, img_width, img, pc_labels, dense_depths, confidences]
            cache_id = image_idx
        else:
            calib, pc_rect, pts_image_2d, img_height, img_width, img, pc_labels, dense_depths, confidences = cache

        if det_box_class_list[box_idx] not in type_whitelist:
            continue

        for augment_i in range(augment_x):
            # 2D BOX: Get pts rect backprojected
            if perturb_box2d and augment_i > 0:
                xmin, ymin, xmax, ymax = random_shift_box2d(det_box_geometry_list[box_idx])
            else:
                xmin, ymin, xmax, ymax = det_box_geometry_list[box_idx]

            box_fov_inds = (pts_image_2d[:, 0] < xmax) & \
                           (pts_image_2d[:, 0] >= xmin) & \
                           (pts_image_2d[:, 1] < ymax) & \
                           (pts_image_2d[:, 1] >= ymin)
            pc_in_box_count = np.count_nonzero(box_fov_inds)

            # Pass objects that are too small
            if ymax - ymin < img_height_threshold or \
                    pc_in_box_count < lidar_point_threshold:
                continue

            # Get frustum angle (according to center pixel in 2D BOX)
            box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
            uvdepth = np.zeros((1, 3))
            uvdepth[0, 0:2] = box2d_center
            uvdepth[0, 2] = 20  # some random depth
            box2d_center_rect = calib.project_image_to_rect(uvdepth)
            frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                            box2d_center_rect[0, 0])

            image_box_detected_label_list[image_idx][box_fov_inds, augment_i] = True
            pts_2d = pts_image_2d[box_fov_inds, :]

            if not use_depth_net:
                pc_in_box_colors = img[pts_2d[:, 1], pts_2d[:, 0], :]
                pc_in_box = np.concatenate((pc_rect[box_fov_inds, :], pc_in_box_colors), axis=1)

                pc_in_box_labels = np.zeros((pc_in_box_count, 1), np.int_)
                pc_in_box_labels[pc_labels[box_fov_inds] == det_box_class_list[box_idx]] = 1
                pc_in_box_labels[pc_labels[box_fov_inds] == 'DontCare'] = -1
            else:
                num_lidar_points_in_box = np.shape(pts_2d)[0]
                if num_lidar_points_in_box >= fill_n_points:
                    pc_in_box_labels = np.zeros((pc_in_box_count, 1), np.int_)
                    pc_in_box_labels[pc_labels[box_fov_inds] == det_box_class_list[box_idx]] = 1
                    pc_in_box_labels[pc_labels[box_fov_inds] == 'DontCare'] = -1

                    selected_pixels_in_box_row = pts_2d[:, 1]
                    selected_pixels_in_box_col = pts_2d[:, 0]
                else:
                    int_x_min = int(max(np.floor(xmin), 0))
                    int_x_max = int(min(np.ceil(xmax), img_width-1))
                    box_x_width = int_x_max-int_x_min+1

                    int_y_min = int(max(np.floor(ymin), 0))
                    int_y_max = int(min(np.ceil(ymax), img_height-1))
                    box_y_width = int_y_max-int_y_min+1

                    num_pixels_in_box = box_x_width * box_y_width

                    labels = np.zeros((box_y_width, box_x_width), np.int_) -1
                    true_inds = np.squeeze(pc_labels[box_fov_inds] == det_box_class_list[box_idx])
                    false_inds = np.logical_and(np.logical_not(true_inds),
                                                np.squeeze(pc_labels[box_fov_inds] != 'DontCare'))
                    labels[pts_2d[true_inds, 1]-int_y_min, pts_2d[true_inds, 0]-int_x_min] = 1
                    labels[pts_2d[false_inds, 1]-int_y_min, pts_2d[false_inds, 0]-int_x_min] = 0

                    box_sub_pixels_row, box_sub_pixels_col = np.indices((box_y_width, box_x_width))
                    box_sub_pixels_row = np.reshape(box_sub_pixels_row, -1)
                    box_sub_pixels_col = np.reshape(box_sub_pixels_col, -1)
                    pixels_in_box_row = box_sub_pixels_row + int_y_min
                    pixels_in_box_col = box_sub_pixels_col + int_x_min

                    if num_pixels_in_box < fill_n_points:
                        selected_box_sub_pixels_row = box_sub_pixels_row
                        selected_box_sub_pixels_col = box_sub_pixels_col
                        selected_pixels_in_box_row = pixels_in_box_row
                        selected_pixels_in_box_col = pixels_in_box_col
                    else:
                        inds_in_box = np.squeeze(np.where(labels[box_sub_pixels_row, box_sub_pixels_col] != -1))
                        other_inds_in_box = np.squeeze(np.where(labels[box_sub_pixels_row, box_sub_pixels_col] == -1))
                        num_points_to_fill = min(fill_n_points, num_pixels_in_box)-num_lidar_points_in_box
                        if from_depth_completion:
                            other_inds_in_box_confidence_order = np.argsort(
                                -confidences[box_sub_pixels_row[other_inds_in_box],
                                box_sub_pixels_col[other_inds_in_box]])
                            most_confident_other_inds = other_inds_in_box[
                                other_inds_in_box_confidence_order[:num_points_to_fill]]
                            sected_other_inds = most_confident_other_inds
                        else: # from_depth_prediction
                            sected_other_inds = np.random.choice(other_inds_in_box, num_points_to_fill, replace=False)

                        selected_inds_in_box = np.concatenate((inds_in_box, sected_other_inds), axis=0)

                        selected_box_sub_pixels_row = box_sub_pixels_row[selected_inds_in_box]
                        selected_box_sub_pixels_col = box_sub_pixels_col[selected_inds_in_box]
                        selected_pixels_in_box_row = pixels_in_box_row[selected_inds_in_box]
                        selected_pixels_in_box_col = pixels_in_box_col[selected_inds_in_box]

                    pc_in_box_labels = labels[selected_box_sub_pixels_row, selected_box_sub_pixels_col]

                depths_in_box = dense_depths[selected_pixels_in_box_row, selected_pixels_in_box_col]
                new_pc_img_in_box = np.concatenate((np.ndarray.astype(np.expand_dims(selected_pixels_in_box_col, 1),
                                                                      np.float),
                                                   np.ndarray.astype(np.expand_dims(selected_pixels_in_box_row, 1),
                                                                     np.float),
                                                   np.expand_dims(depths_in_box, 1)), axis=1)
                new_pc_rect_in_box = calib.project_image_to_rect(new_pc_img_in_box)
                pc_in_box_colors = img[selected_pixels_in_box_row, selected_pixels_in_box_col, :]

                if from_depth_completion:
                    confidences_in_box = np.expand_dims(
                        confidences[selected_pixels_in_box_row, selected_pixels_in_box_col], 1)
                    pc_in_box = np.concatenate((new_pc_rect_in_box, confidences_in_box, pc_in_box_colors), axis=1)
                else: #from_depth_prediction
                    pc_in_box = np.concatenate((new_pc_rect_in_box, pc_in_box_colors), axis=1)

            box_class_list.append(det_box_class_list[box_idx])
            box_certainty_list.append(det_box_certainty_list[box_idx])
            pc_in_box_list.append(pc_in_box)
            frustum_angle_list.append(frustum_angle)
            pc_in_box_label_list.append(pc_in_box_labels)

            pc_in_box_inds_list.append(box_fov_inds)
            box_image_id_list.append(image_idx)

    fn = np.zeros((len(type_whitelist)), np.int_)
    for image_idx in image_idx_list:
        for augment_i in range(augment_x):
            undetected_labels = image_pc_label_list[image_idx][
                np.logical_not(image_box_detected_label_list[image_idx][:, augment_i])]
            for type_idx in range(len(type_whitelist)):
                fn += np.count_nonzero(undetected_labels == type_whitelist[type_idx])

    with open(output_filename, 'wb') as fp:
        # box lists
        pickle.dump(box_class_list, fp)
        pickle.dump(box_certainty_list, fp)
        pickle.dump(pc_in_box_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(pc_in_box_label_list, fp)
        pickle.dump(fn, fp)
        #   for labeled images
        pickle.dump(box_image_id_list, fp)
        pickle.dump(pc_in_box_inds_list, fp)
        pickle.dump(image_pc_label_list, fp)


    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = pc_in_box_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
                              fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:, 0], p1[:, 1], p1[:, 2], p1[:, 1], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
                              fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:, 2], -p1[:, 0], -p1[:, 1], seg, mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
            input()


def compare_source(split_file_datapath, split,
                   type_whitelist=['Car'],
                   from_rgb_detection=True,
                   rgb_det_filename="",
                   img_height_threshold=25,
                   lidar_point_threshold=5,
                   from_unguided_depth_completion=False,
                   from_guided_depth_completion=False,
                   from_depth_prediction=False):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        split_file_datapath: string, each line of the file is a image sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augment_x: scalar, how many augmentations to have for each 2D box (no augmentation => 1).
        rgb_det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''

    from_depth_completion = from_unguided_depth_completion or from_guided_depth_completion
    use_depth_net = from_depth_completion or from_depth_prediction
    assert int(from_guided_depth_completion) + int(from_unguided_depth_completion) + int(from_depth_prediction) <= 1
    assert use_depth_net

    if from_depth_completion:
        if from_guided_depth_completion:
            bla = 0
            # depth_net = load_net('exp_guided_nconv_cnn_l1', mode='bla', checkpoint_num=40, set_='bla')
        else: # from_unguided_depth_completion:
            sys.path.append(os.path.join(ROOT_DIR, '../nconv'))
            from run_nconv_cnn import load_net
            depth_net = load_net('exp_unguided_depth', mode='bla', checkpoint_num=3, set_='bla')
        desired_image_height = 352
        desired_image_width = 1216
    elif from_depth_prediction:
        sys.path.append(os.path.join(ROOT_DIR, '../monodepth2'))
        from monodepth_external import load_net
        depth_net = load_net("mono+stereo_1024x320", use_cuda=True)

    image_idx_list = [int(line.rstrip()) for line in open(split_file_datapath)]
    dataset = kitti_object(os.path.join(ROOT_DIR, './../../data/kitti_object'), split)

    # image labels
    image_pc_label_list = []
    image_box_detected_label_list = []

    if not from_rgb_detection:
        det_box_image_index_list = []
        det_box_class_list = []
        det_box_geometry_list = []
        det_box_certainty_list = []

    o_filler = np.zeros(0, np.object)
    b_filler = np.zeros(0, np.bool_)
    for image_idx in range(dataset.num_samples):
        image_pc_label_list.append(o_filler)
        image_box_detected_label_list.append(b_filler)

    for image_idx in image_idx_list:
        print('image idx: %d/%d' % (image_idx, dataset.num_samples))

        calib = dataset.get_calibration(image_idx)  # 3 by 4 matrix

        pc_velo = dataset.get_lidar(image_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]

        img = dataset.get_image(image_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        _, _, img_fov_inds, _ = get_lidar_in_image_fov( \
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
        pc_rect = pc_rect[img_fov_inds, :]

        label_objects = dataset.get_label_objects(image_idx)
        pc_labels = np.zeros((np.size(pc_rect, 0)), np.object)
        for label_object in label_objects:
            _, box3d_pts_3d = utils.compute_box_3d(label_object, calib.P)
            _, instance_pc_indexes = extract_pc_in_box3d(pc_rect, box3d_pts_3d)
            overlapping_3d_boxes = np.nonzero(pc_labels[instance_pc_indexes])[0]
            pc_labels[instance_pc_indexes] = label_object.type
            (pc_labels[instance_pc_indexes])[overlapping_3d_boxes] = 'DontCare'
            if not from_rgb_detection and label_object.type in type_whitelist:
                det_box_geometry_list.append(label_object.box2d)
                det_box_certainty_list.append(1)
                det_box_class_list.append(label_object.type)
                det_box_image_index_list.append(image_idx)

        image_pc_label_list[image_idx] = pc_labels
        image_box_detected_label_list[image_idx] = np.zeros(pc_labels.shape[0], np.bool_)

    if from_rgb_detection:
        det_box_image_index_list, det_box_class_list, det_box_geometry_list, det_box_certainty_list = \
            read_box_file(rgb_det_filename)

    cache_id = -1
    cache = None

    box_class_list = []
    pc_in_box_label_list = []
    alt_pc_in_box_label_list = []
    box_image_id_list = []

    for box_idx in range(len(det_box_image_index_list)):
        image_idx = det_box_image_index_list[box_idx]
        print('box idx: %d/%d, image idx: %d' % \
              (box_idx, len(det_box_image_index_list), image_idx))
        if cache_id != image_idx:
            calib = dataset.get_calibration(image_idx)  # 3 by 4 matrix

            pc_velo = dataset.get_lidar(image_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
            pc_rect[:, 3] = pc_velo[:, 3]

            img = dataset.get_image(image_idx)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width, img_channel = img.shape

            _, pts_image_2d, img_fov_inds, pc_image_depths = get_lidar_in_image_fov( \
                pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
            pc_rect = pc_rect[img_fov_inds, :]
            pts_image_2d = np.ndarray.astype(np.round(pts_image_2d[img_fov_inds, :]), int)
            pts_image_2d[pts_image_2d < 0] = 0
            pts_image_2d[pts_image_2d[:, 0] >= img_width, 0] = img_width-1
            pts_image_2d[pts_image_2d[:, 1] >= img_height, 1] = img_height-1
            pc_labels = image_pc_label_list[image_idx]

            dense_depths = []
            if from_unguided_depth_completion:
                lidarmap = dataset.generate_depth_map(image_idx, 2, desired_image_width, desired_image_height)
                rgb = Image.fromarray(img).resize((desired_image_width, desired_image_height), Image.LANCZOS)
                rgb = np.array(rgb, dtype=np.float16)
                dense_depths, confidences = depth_net.return_one_prediction(lidarmap*255, rgb, img_width, img_height)
            elif from_guided_depth_completion:
                res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/completed_depth')
                (dense_depths, confidences) = np.load(os.path.join(res_dir, str(image_idx)+'.npy'))
            elif from_depth_prediction:
                dense_depths = depth_net.return_one_prediction(img, post_process=False)

            depths_in_box = dense_depths[pts_image_2d[:, 1], pts_image_2d[:, 0]]
            alt_pc = np.concatenate((np.ndarray.astype(np.expand_dims(pts_image_2d[:, 0], 1), np.float),
                                      np.ndarray.astype(np.expand_dims(pts_image_2d[:, 1], 1), np.float),
                                      np.expand_dims(depths_in_box, 1)), axis=1)

            alt_pc_labels = np.zeros((np.size(pc_rect, 0)), np.object)
            for label_object in dataset.get_label_objects(image_idx):
                _, box3d_pts_3d = utils.compute_box_3d(label_object, calib.P)
                _, instance_pc_indexes = extract_pc_in_box3d(alt_pc, box3d_pts_3d)
                overlapping_3d_boxes = np.nonzero(pc_labels[instance_pc_indexes])[0]
                alt_pc_labels[instance_pc_indexes] = label_object.type
                (alt_pc_labels[instance_pc_indexes])[overlapping_3d_boxes] = 'DontCare'

            cache = [pts_image_2d, pc_labels, alt_pc_labels]
            cache_id = image_idx
        else:
            pts_image_2d, pc_labels, alt_pc_labels = cache


        if det_box_class_list[box_idx] not in type_whitelist:
            continue

        xmin, ymin, xmax, ymax = det_box_geometry_list[box_idx]

        box_fov_inds = (pts_image_2d[:, 0] < xmax) & \
                       (pts_image_2d[:, 0] >= xmin) & \
                       (pts_image_2d[:, 1] < ymax) & \
                       (pts_image_2d[:, 1] >= ymin)
        pc_in_box_count = np.count_nonzero(box_fov_inds)

        # Pass objects that are too small
        if ymax - ymin < img_height_threshold or \
                pc_in_box_count < lidar_point_threshold:
            continue

        image_box_detected_label_list[image_idx][box_fov_inds] = True

        pc_in_box_labels = np.zeros((pc_in_box_count), np.int_)
        pc_in_box_labels[pc_labels[box_fov_inds] == det_box_class_list[box_idx]] = 1
        pc_in_box_labels[pc_labels[box_fov_inds] == 'DontCare'] = -1

        alt_pc_in_box_labels = np.zeros((pc_in_box_count), np.int_)
        alt_pc_in_box_labels[pc_labels[box_fov_inds] == det_box_class_list[box_idx]] = 1
        alt_pc_in_box_labels[pc_labels[box_fov_inds] == 'DontCare'] = -1

        box_class_list.append(det_box_class_list[box_idx])
        pc_in_box_label_list.append(pc_in_box_labels)
        alt_pc_in_box_label_list.append(alt_pc_in_box_labels)
        box_image_id_list.append(image_idx)

    fn = np.zeros((len(type_whitelist)), np.int_)
    for image_idx in image_idx_list:
        undetected_labels = image_pc_label_list[image_idx][
            np.logical_not(image_box_detected_label_list[image_idx][:])]
        for type_idx in range(len(type_whitelist)):
            fn += np.count_nonzero(undetected_labels == type_whitelist[type_idx])

    # evaluate
    NUM_CLASSES = 2
    NUM_REAL_CLASSES = len(type_whitelist)

    # To collect statistics
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    epsilon = 1e-12

    # box class-level metrics
    tp_sum = np.zeros(NUM_REAL_CLASSES)
    box_fn_sum = np.zeros(NUM_REAL_CLASSES)
    fp_sum = np.zeros(NUM_REAL_CLASSES)
    tn_sum = np.zeros(NUM_REAL_CLASSES)

    # box instance-level metrics
    iiou_sum = np.zeros(NUM_REAL_CLASSES)
    ire_sum = np.zeros(NUM_REAL_CLASSES)
    ipr_sum = np.zeros(NUM_REAL_CLASSES)
    i_sum = np.zeros(NUM_REAL_CLASSES)

    for box_idx in range(len(pc_in_box_label_list)):
        labels = pc_in_box_label_list[box_idx]
        alt_labels = alt_pc_in_box_label_list[box_idx]
        box_class = type_whitelist.index(box_class_list[box_idx])

        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(labels == l)
            total_correct_class[l] += (np.sum((alt_labels == l) & (labels == l)))

        icare = labels != -1

        tps = np.sum(labels * alt_labels * icare)
        fns = np.sum(labels * (1 - alt_labels) * icare)
        fps = np.sum((1 - labels) * alt_labels * icare)
        tns = np.sum((1 - labels) * (1 - alt_labels) * icare)

        iiou = tps.astype(np.float) / (tps + fns + fps + epsilon)
        ipr = tps.astype(np.float) / (tps + fps + epsilon)
        ire = tps.astype(np.float) / (tps + fns + epsilon)

        iiou_sum[box_class] += iiou
        ire_sum[box_class] += ire
        ipr_sum[box_class] += ipr
        i_sum[box_class] += 1

        tp_sum[box_class] += tps
        box_fn_sum[box_class] += fns
        fp_sum[box_class] += fps
        tn_sum[box_class] += tns

    print('eval mean loss: %f' % (loss_sum / np.sum(i_sum)))
    print('eval segmentation accuracy: %f' % (float(np.sum(tp_sum + tn_sum)) /
                                              (float(np.sum(tp_sum + box_fn_sum + fp_sum + tn_sum)) + epsilon)))
    print('eval segmentation avg class acc: %f' % np.mean((tp_sum + tn_sum).astype(np.float)
                                                               / (tp_sum + box_fn_sum + fp_sum + tn_sum + epsilon)))

    box_ious = tp_sum.astype(np.float) / (tp_sum + box_fn_sum + fp_sum + epsilon)
    box_prs = tp_sum.astype(np.float) / (tp_sum + fp_sum + epsilon)
    box_res = tp_sum.astype(np.float) / (tp_sum + box_fn_sum + epsilon)

    box_any_ious = np.sum(tp_sum).astype(np.float) / (np.sum(tp_sum + box_fn_sum + fp_sum) + epsilon)
    box_any_prs = np.sum(tp_sum).astype(np.float) / (np.sum(tp_sum + fp_sum) + epsilon)
    box_any_res = np.sum(tp_sum).astype(np.float) / (np.sum(tp_sum + box_fn_sum) + epsilon)

    iious = iiou_sum.astype(np.float) / (i_sum + epsilon)
    iprs = ipr_sum.astype(np.float) / (i_sum + epsilon)
    ires = ire_sum.astype(np.float) / (i_sum + epsilon)

    iious_any = np.sum(iiou_sum).astype(np.float) / (np.sum(i_sum) + epsilon)
    iprs_any = np.sum(ipr_sum).astype(np.float) / (np.sum(i_sum) + epsilon)
    ires_any = np.sum(ire_sum).astype(np.float) / (np.sum(i_sum) + epsilon)

    # image class-level metrics
    image_fn_sum = box_fn_sum + fn
    image_ious = tp_sum.astype(np.float) / (tp_sum + image_fn_sum + fp_sum + epsilon)
    image_prs = box_prs
    image_res = tp_sum.astype(np.float) / (tp_sum + image_fn_sum + epsilon)

    image_any_ious = np.sum(tp_sum).astype(np.float) / (np.sum(tp_sum + image_fn_sum + fp_sum) + epsilon)
    image_any_prs = box_any_prs
    image_any_res = np.sum(tp_sum).astype(np.float) / (np.sum(tp_sum + image_fn_sum) + epsilon)

    for class_idx in range(NUM_REAL_CLASSES):
        print()
        print(type_whitelist[class_idx])
        print('Box_IOU %d' % box_ious[class_idx])
        print('Box_Precision %d' % box_prs[class_idx])
        print('Box_Recall %d' % box_res[class_idx])
        print('Box_iIOU %d' % iious[class_idx])
        print('Box_iPrecision %d' % iprs[class_idx])
        print('Box_iRecall %d' % ires[class_idx])
        print('Image_IOU %d' % image_ious[class_idx])
        print('Image_Precision %d' % image_prs[class_idx])
        print('Image_Recall %d' % image_res[class_idx])

    print()
    print('Mean Class')
    print('Box_IOU %d' % np.mean(box_ious))
    print('Box_Precision %d' % np.mean(box_prs))
    print('Box_Recall %d' % np.mean(box_res))
    print('Box_iIOU %d' % np.mean(iious))
    print('Box_iPrecision %d' % np.mean(iprs))
    print('Box_iRecall %d' % np.mean(ires))
    print('Image_IOU %d' % np.mean(image_ious))
    print('Image_Precision %d' % np.mean(image_prs))
    print('Image_Recall %d' % np.mean(image_res))

    print()
    print('Class Any')
    print('Box_IOU %d' % np.mean(box_any_ious))
    print('Box_Precision %d' % np.mean(box_any_prs))
    print('Box_Recall %d' % np.mean(box_any_res))
    print('Box_iIOU %d' % np.mean(iious_any))
    print('Box_iPrecision %d' % np.mean(iprs_any))
    print('Box_iRecall %d' % np.mean(ires_any))
    print('Image_IOU %d' % np.mean(image_any_ious))
    print('Image_Precision %d' % np.mean(image_any_prs))
    print('Image_Recall %d' % np.mean(image_any_res))


def write_2d_rgb_detection(det_filename, split, result_dir):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format. 
        
    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        result_dir: string, folder path for results dumping
    Output:
        None (will write <xxx>.txt files to disk)

    Usage:
        write_2d_rgb_detection("val_det.txt", "training", "results")
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, './../../data/kitti_object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_box_file(det_filename)
    # map from idx to list of strings, each string is a line without \n
    results = {} 
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close() 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true',
                        help='Run demo.')
    parser.add_argument('--gen_train', action='store_true',
                        help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_train_rgb_detection', action='store_true',
                        help='Generate train split frustum data with RGB detection 2D boxes')
    parser.add_argument('--gen_val', action='store_true',
                        help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection', action='store_true',
                        help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--show_pixel_statistics', action='store_true',
                        help='Show Pixel Statistics')
    parser.add_argument('--show_alt_depth_source_seg_statistics', action='store_true',
                        help='Show Segmentation Statistics with ideal boxes and alternate depth source')
    parser.add_argument('--car_only', action='store_true',
                        help='Only generate cars; otherwise cars, peds and cycs')
    parser.add_argument('--from_unguided_depth_completion', action='store_true',
                        help='Use point cloud from unguided depth completion')
    parser.add_argument('--from_guided_depth_completion', action='store_true',
                        help='Use point cloud from guided depth completion')
    parser.add_argument('--from_depth_prediction', action='store_true',
                        help='Use point cloud from depth prediction')
    parser.add_argument('--fill_n_points', type=int, default=-1,
                        help='Fill x points with depth completion / prediction, -1 = use all')
    args = parser.parse_args()

    if args.demo:
        demo()
        exit()

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.from_unguided_depth_completion:
        output_prefix += 'unguided_completion_'
    if args.from_guided_depth_completion:
        output_prefix += 'guided_completion_'
    if args.from_depth_prediction:
        output_prefix += 'prediction_'

    if not args.show_alt_depth_source_seg_statistics:
        if args.gen_val:
            extract_frustum_data(
                os.path.join(BASE_DIR, 'image_sets/val.txt'),
                'training',
                os.path.join(BASE_DIR, output_prefix+'val.pickle'),
                viz=False, perturb_box2d=False, augment_x=1,
                type_whitelist=type_whitelist,
                from_guided_depth_completion=args.from_guided_depth_completion,
                from_unguided_depth_completion=args.from_unguided_depth_completion,
                from_depth_prediction=args.from_depth_prediction,
                from_rgb_detection=False,
                fill_n_points=args.fill_n_points)

        if args.gen_val_rgb_detection:
            extract_frustum_data(
                os.path.join(BASE_DIR, 'image_sets/val.txt'),
                'training',
                os.path.join(BASE_DIR, output_prefix+'val_rgb_detection.pickle'),
                viz=False, perturb_box2d=False, augment_x=1,
                rgb_det_filename=os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
                type_whitelist=type_whitelist,
                from_rgb_detection=True,
                from_guided_depth_completion=args.from_guided_depth_completion,
                from_unguided_depth_completion=args.from_unguided_depth_completion,
                from_depth_prediction=args.from_depth_prediction,
                fill_n_points=args.fill_n_points)

        if args.gen_train:
            extract_frustum_data(
                os.path.join(BASE_DIR, 'image_sets/train.txt'),
                'training',
                os.path.join(BASE_DIR, output_prefix+'train.pickle'),
                viz=False, perturb_box2d=True, augment_x=5,
                type_whitelist=type_whitelist,
                from_guided_depth_completion=args.from_guided_depth_completion,
                from_unguided_depth_completion=args.from_unguided_depth_completion,
                from_depth_prediction=args.from_depth_prediction,
                from_rgb_detection=False,
                fill_n_points=args.fill_n_points)

        if args.gen_train_rgb_detection:
            extract_frustum_data(
                os.path.join(BASE_DIR, 'image_sets/train.txt'),
                'training',
                os.path.join(BASE_DIR, output_prefix+'train_rgb_detection.pickle'),
                viz=False, perturb_box2d=False, augment_x=1,
                rgb_det_filename=os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_train.txt'),
                type_whitelist=type_whitelist,
                from_rgb_detection=True,
                from_guided_depth_completion=args.from_guided_depth_completion,
                from_unguided_depth_completion=args.from_unguided_depth_completion,
                from_depth_prediction=args.from_depth_prediction,
                fill_n_points=args.fill_n_points)

    if args.show_pixel_statistics:
        show_points_per_box_statistics(
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            'rgb_detection val',
            rgb_det_filename=os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
            type_whitelist=type_whitelist,
            from_rgb_detection=True)
        show_points_per_box_statistics(
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            'ideal boxes val',
            rgb_det_filename=os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
            type_whitelist=type_whitelist,
            from_rgb_detection=False)
        show_points_per_box_statistics(
            os.path.join(BASE_DIR, 'image_sets/train.txt'),
            'training',
            'rgb_detection train',
            rgb_det_filename=os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_train.txt'),
            type_whitelist=type_whitelist,
            from_rgb_detection=True)
        show_points_per_box_statistics(
            os.path.join(BASE_DIR, 'image_sets/train.txt'),
            'training',
            'ideal boxes train',
            rgb_det_filename=os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_train.txt'),
            type_whitelist=type_whitelist,
            from_rgb_detection=False)

    if args.show_alt_depth_source_seg_statistics:
        if args.gen_val:
            compare_source(
                os.path.join(BASE_DIR, 'image_sets/val.txt'),
                'training',
                type_whitelist=type_whitelist,
                from_guided_depth_completion=args.from_guided_depth_completion,
                from_unguided_depth_completion=args.from_unguided_depth_completion,
                from_depth_prediction=args.from_depth_prediction,
                from_rgb_detection=False)

        if args.gen_val_rgb_detection:
            compare_source(
                os.path.join(BASE_DIR, 'image_sets/val.txt'),
                'training',
                rgb_det_filename=os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
                type_whitelist=type_whitelist,
                from_rgb_detection=True,
                from_guided_depth_completion=args.from_guided_depth_completion,
                from_unguided_depth_completion=args.from_unguided_depth_completion,
                from_depth_prediction=args.from_depth_prediction)

        if args.gen_train:
            compare_source(
                os.path.join(BASE_DIR, 'image_sets/train.txt'),
                'training',
                type_whitelist=type_whitelist,
                from_guided_depth_completion=args.from_guided_depth_completion,
                from_unguided_depth_completion=args.from_unguided_depth_completion,
                from_depth_prediction=args.from_depth_prediction,
                from_rgb_detection=False)

        if args.gen_train_rgb_detection:
            compare_source(
                os.path.join(BASE_DIR, 'image_sets/train.txt'),
                'training',
                rgb_det_filename=os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_train.txt'),
                type_whitelist=type_whitelist,
                from_rgb_detection=True,
                from_guided_depth_completion=args.from_guided_depth_completion,
                from_unguided_depth_completion=args.from_unguided_depth_completion,
                from_depth_prediction=args.from_depth_prediction)
