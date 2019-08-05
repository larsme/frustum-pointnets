''' Util functions for training and evaluation.

Author: Charles R. Qi
Date: September 2017
'''

import numpy as np


# def get_batch(dataset, batch_box_idxs, start_idx, end_idx,
#               num_point, num_channel, num_real_classes, segment_all_points=False):
#     ''' Prepare batch data for training/evaluation.
#     batch size is determined by start_idx-end_idx
#
#     Input:
#         dataset: an instance of FrustumDataset class
#         batch_box_idxs: a list of data element indices
#         start_idx: int scalar, start position in batch_box_idxs
#         end_idx: int scalar, end position in batch_box_idxs
#         num_point: int scalar
#         num_channel: int scalar
#         from_rgb_detection: bool
#     Output:
#         batched data and label
#     '''
    # if dataset.from_rgb_detection:
    #     return get_batch_from_rgb_detection(dataset, batch_box_idxs, start_idx, end_idx,
    #                                         num_point, num_channel, num_real_classes, segment_all_points)

    # bsize = end_idx-start_idx
    # batch_input_pc = np.zeros((bsize, num_point, num_channel))
    # batch_gt_labels = np.zeros((bsize, num_point), dtype=np.bool)
    # # batch_center = np.zeros((bsize, 3))
    # # batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    # # batch_heading_residual = np.zeros((bsize,))
    # # batch_size_class = np.zeros((bsize,), dtype=np.int32)
    # # batch_size_residual = np.zeros((bsize, 3))
    # batch_rot_angle = np.zeros((bsize,))
    # if dataset.box_class_one_hot:
    #     batch_one_hot_vec = np.zeros((bsize, num_real_classes), dtype=np.bool) # for car,ped,cyc
    # for i in range(bsize):
    #     if dataset.box_class_one_hot:
    #         ps, seg, center, hclass, hres, sclass, sres, rotangle, onehotvec = \
    #             dataset[batch_box_idxs[i+start_idx]]
    #         batch_one_hot_vec[i] = onehotvec
    #     else:
    #         ps, seg, center, hclass, hres, sclass, sres, rotangle = \
    #             dataset[batch_box_idxs[i+start_idx]]
    #     batch_input_pc[i, ...] = ps[:,0:num_channel]
    #     batch_gt_labels[i, :] = seg
    #     # batch_center[i,:] = center
    #     # batch_heading_class[i] = hclass
    #     # batch_heading_residual[i] = hres
    #     # batch_size_class[i] = sclass
    #     # batch_size_residual[i] = sres
    #     batch_rot_angle[i] = rotangle
    # # if dataset.one_hot:
    # #     return batch_input_pc, batch_gt_labels, batch_center, \
    # #         batch_heading_class, batch_heading_residual, \
    # #         batch_size_class, batch_size_residual, \
    # #         batch_rot_angle, batch_one_hot_vec
    # # else:
    # #     return batch_input_pc, batch_gt_labels, batch_center, \
    # #         batch_heading_class, batch_heading_residual, \
    # #         batch_size_class, batch_size_residual, batch_rot_angle
    # if dataset.box_class_one_hot:
    #     return batch_input_pc, batch_gt_labels, batch_rot_angle, batch_one_hot_vec
    # else:
    #     return batch_input_pc, batch_gt_labels, batch_rot_angle


# def get_batch_from_rgb_detection(dataset, batch_box_idxs, start_idx, end_idx,
#                                  num_point, num_channel):
#     bsize = end_idx-start_idx
#     batch_data = np.zeros((bsize, num_point, num_channel))
#     batch_rot_angle = np.zeros((bsize,))
#     batch_prob = np.zeros((bsize,))
#     if dataset.one_hot:
#         batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
#     for i in range(bsize):
#         if dataset.one_hot:
#             ps,rotangle,prob,onehotvec = dataset[batch_box_idxs[i+start_idx]]
#             batch_one_hot_vec[i] = onehotvec
#         else:
#             ps,rotangle,prob = dataset[batch_box_idxs[i+start_idx]]
#         batch_data[i,...] = ps[:,0:num_channel]
#         batch_rot_angle[i] = rotangle
#         batch_prob[i] = prob
#     if dataset.one_hot:
#         return batch_data, batch_rot_angle, batch_prob, batch_one_hot_vec
#     else:
#         return batch_data, batch_rot_angle, batch_prob


def get_batch(dataset, fixed_batch_size, batch_box_idxs, num_point, num_channel, num_real_classes,
              batch_box_points_left_to_sample=None):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        fixed_batch_size: fixed dimension of outputs
        batch_box_idxs: a list of data element indices
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    real_batch_size = len(batch_box_idxs)
    batch_input_pc = np.zeros((fixed_batch_size, num_point, num_channel))
    batch_gt_labels = np.zeros((fixed_batch_size, num_point))
    batch_rot_angle = np.zeros((fixed_batch_size,))
    batch_box_certainty = np.zeros((fixed_batch_size,))

    if dataset.segment_all_points and batch_box_points_left_to_sample is None:
        for i in range(real_batch_size):
            batch_box_points_left_to_sample.append([])
    new_batch_box_points_left_to_sample = []

    if dataset.box_class_one_hot:
        batch_one_hot_vec = np.zeros((fixed_batch_size, num_real_classes), np.bool_) # for car,ped,cyc

    for i in range(real_batch_size):
        if dataset.segment_all_points:
            data = dataset[batch_box_idxs[i], batch_box_points_left_to_sample[i]]
            new_batch_box_points_left_to_sample.append(data[4])
        else:
            data = dataset[batch_box_idxs[i], None]

        batch_input_pc[i, ...] = data[0]
        batch_gt_labels[i, ...] = data[1]
        batch_rot_angle[i] = data[2]
        batch_box_certainty[i] = data[3]

        if dataset.box_class_one_hot:
            batch_one_hot_vec[i, ...] = data[5]

    if dataset.box_class_one_hot:
        return batch_input_pc, batch_gt_labels, batch_rot_angle, batch_box_certainty, \
               new_batch_box_points_left_to_sample, batch_one_hot_vec
    else:
        return batch_input_pc, batch_gt_labels, batch_rot_angle, batch_box_certainty, \
               new_batch_box_points_left_to_sample


