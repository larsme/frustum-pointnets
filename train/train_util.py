''' Util functions for training and evaluation.

Author: Charles R. Qi
Date: September 2017
'''

import numpy as np


def get_batch(dataset, idxs, start_idx, end_idx,
              num_point, num_channel, num_real_classes):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if dataset.from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
                                            num_point, num_channel, num_real_classes)

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
    #             dataset[idxs[i+start_idx]]
    #         batch_one_hot_vec[i] = onehotvec
    #     else:
    #         ps, seg, center, hclass, hres, sclass, sres, rotangle = \
    #             dataset[idxs[i+start_idx]]
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


# def get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
#                                  num_point, num_channel):
#     bsize = end_idx-start_idx
#     batch_data = np.zeros((bsize, num_point, num_channel))
#     batch_rot_angle = np.zeros((bsize,))
#     batch_prob = np.zeros((bsize,))
#     if dataset.one_hot:
#         batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
#     for i in range(bsize):
#         if dataset.one_hot:
#             ps,rotangle,prob,onehotvec = dataset[idxs[i+start_idx]]
#             batch_one_hot_vec[i] = onehotvec
#         else:
#             ps,rotangle,prob = dataset[idxs[i+start_idx]]
#         batch_data[i,...] = ps[:,0:num_channel]
#         batch_rot_angle[i] = rotangle
#         batch_prob[i] = prob
#     if dataset.one_hot:
#         return batch_data, batch_rot_angle, batch_prob, batch_one_hot_vec
#     else:
#         return batch_data, batch_rot_angle, batch_prob


def get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx, num_point, num_channel, num_real_classes):
    bsize = end_idx-start_idx
    batch_input_pc = np.zeros((bsize, num_point, num_channel))
    batch_gt_labels = np.zeros((bsize, num_point))
    batch_rot_angle = np.zeros((bsize,))
    batch_box_certainty = np.zeros((bsize,))
    if dataset.box_class_one_hot:
        batch_one_hot_vec = np.zeros((bsize, num_real_classes), np.bool_) # for car,ped,cyc
        for i in range(bsize):
            batch_input_pc[i, ...], batch_gt_labels[i, ...], batch_rot_angle[i], \
                batch_box_certainty[i], batch_one_hot_vec[i, ...] \
                = dataset[idxs[i + start_idx]]
        return batch_input_pc, batch_gt_labels, batch_rot_angle, batch_box_certainty, batch_one_hot_vec
    else:
        for i in range(bsize):
            batch_input_pc[i, ...], batch_gt_labels[i, ...], batch_rot_angle[i], batch_box_certainty[i] \
                = dataset[idxs[i + start_idx]]
        return batch_input_pc, batch_gt_labels, batch_rot_angle, batch_box_certainty


