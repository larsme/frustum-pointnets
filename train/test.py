''' Evaluating Frustum PointNets.
Write evaluation results to KITTI format labels.
and [optionally] write results to pickle files.

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
import pickle as pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR in sys.path:
    sys.path.remove(BASE_DIR)
sys.path.append(ROOT_DIR)
from models.model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import train.provider as provider
from train.train_util import get_batch

NUM_CLASSES = 2
NUM_REAL_CLASSES = 3
REAL_CLASSES = ['Car', 'Pedestrian', 'Cyclist']
FLAGS = BATCH_SIZE = NUM_POINT = GPU_INDEX = NUM_CHANNEL = TEST_DATASET = MODEL_PATH = MODEL = 0


def get_session_and_ops():
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            # pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            # heading_class_label_pl, heading_residual_label_pl, \
            # size_class_label_pl, size_residual_label_pl = \
            #     MODEL.placeholder_inputs(batch_size, num_point)
            batch_input_pc, batch_box_certainty, batch_gt_labels, batch_one_hot_vec = \
                MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_CHANNEL, NUM_REAL_CLASSES)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            # end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
            #     is_training_pl)

            # logits for element or no element (not just 1 prob)
            batch_box_label_prob = tf.to_float(batch_one_hot_vec) *\
                                   tf.tile(tf.expand_dims(batch_box_certainty, axis=1 ), [1, NUM_REAL_CLASSES])
            pc_pred_logits = MODEL.get_model(batch_input_pc, batch_box_label_prob, is_training_pl)
            # loss = MODEL.get_loss(labels_pl, centers_pl,
            #     heading_class_label_pl, heading_residual_label_pl,
            #     size_class_label_pl, size_residual_label_pl, end_points)

            icare = tf.cast(tf.not_equal(batch_gt_labels, -1), tf.int32, 'icare')
            loss = MODEL.get_loss(batch_gt_labels, pc_pred_logits, icare)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        # ops = {'pointclouds_pl': pointclouds_pl,
        #        'one_hot_vec_pl': one_hot_vec_pl,
        #        'labels_pl': labels_pl,
        #        'centers_pl': centers_pl,
        #        'heading_class_label_pl': heading_class_label_pl,
        #        'heading_residual_label_pl': heading_residual_label_pl,
        #        'size_class_label_pl': size_class_label_pl,
        #        'size_residual_label_pl': size_residual_label_pl,
        #        'is_training_pl': is_training_pl,
        #        'logits': end_points['mask_logits'],
        #        'center': end_points['center'],
        #        'end_points': end_points,
        #        'loss': loss}

        ops = {'batch_input_pc': batch_input_pc,
               'batch_one_hot_vec': batch_one_hot_vec,
               'batch_box_certainty': batch_box_certainty,
               'batch_gt_labels': batch_gt_labels,
               'is_training_pl': is_training_pl,
               'pc_pred_logits': pc_pred_logits,
               'loss': loss}
        return sess, ops


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs


def inference(sess, ops, input_pcs, one_hot_vects, box_certainties, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    assert input_pcs.shape[0] % batch_size == 0
    num_batches = int(input_pcs.shape[0] / batch_size)
    inferred_logits = np.zeros((input_pcs.shape[0], input_pcs.shape[1], NUM_CLASSES))
    # centers = np.zeros((input_pcs.shape[0], 3))
    # heading_logits = np.zeros((input_pcs.shape[0], NUM_HEADING_BIN))
    # heading_residuals = np.zeros((input_pcs.shape[0], NUM_HEADING_BIN))
    # size_logits = np.zeros((input_pcs.shape[0], NUM_SIZE_CLUSTER))
    # size_residuals = np.zeros((input_pcs.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((input_pcs.shape[0],)) # 3D box score
    #
    # inferred_logits = ops['pc_pred_logits']
    for i in range(num_batches):
        feed_dict = {\
            ops['batch_input_pc']: input_pcs[i * batch_size:(i + 1) * batch_size, ...],
            ops['batch_one_hot_vec']: one_hot_vects[i * batch_size:(i + 1) * batch_size, :],
            ops['batch_box_certainty']: box_certainties[i * batch_size:(i + 1) * batch_size],
            ops['is_training_pl']: False}

        # batch_logits, batch_centers, \
        # batch_heading_scores, batch_heading_residuals, \
        # batch_size_scores, batch_size_residuals = \
        #     sess.run([ops['pc_pred_logits'], ops['center'],
        #         ep['heading_scores'], ep['heading_residuals'],
        #         ep['size_scores'], ep['size_residuals']],
        #         feed_dict=feed_dict)
        batch_pc_pred_logits = \
            sess.run([ops['pc_pred_logits']],
                feed_dict=feed_dict)
        batch_pc_pred_logits = np.squeeze(batch_pc_pred_logits)

        inferred_logits[i*batch_size:(i+1)*batch_size, ...] = batch_pc_pred_logits
        # centers[i*batch_size:(i+1)*batch_size,...] = batch_centers
        # heading_logits[i*batch_size:(i+1)*batch_size,...] = batch_heading_scores
        # heading_residuals[i*batch_size:(i+1)*batch_size,...] = batch_heading_residuals
        # size_logits[i*batch_size:(i+1)*batch_size,...] = batch_size_scores
        # size_residuals[i*batch_size:(i+1)*batch_size,...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_pc_pred_logits)[:, :, 1] # BxN
        batch_seg_mask = np.argmax(batch_pc_pred_logits, 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1) # B,
        # heading_prob = np.max(softmax(batch_heading_scores),1) # B
        # size_prob = np.max(softmax(batch_size_scores),1) # B,
        # batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        batch_scores = np.log(mask_mean_prob)
        scores[i*batch_size:(i+1)*batch_size] = batch_scores
        # Finished computing scores

    # heading_cls = np.argmax(heading_logits, 1) # B
    # size_cls = np.argmax(size_logits, 1) # B
    # heading_res = np.array([heading_residuals[i,heading_cls[i]] \
    #     for i in range(input_pcs.shape[0])])
    # size_res = np.vstack([size_residuals[i,size_cls[i],:] \
    #     for i in range(input_pcs.shape[0])])

    # return np.argmax(pc_pred_logits, 2), centers, heading_cls, heading_res, \
    #     size_cls, size_res, scores
    return np.argmax(inferred_logits, 2), scores


# def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list, \
#                             heading_cls_list, heading_res_list, \
#                             size_cls_list, size_res_list, \
#                             rot_angle_list, score_list):
# def write_detection_results(result_dir, id_list, type_list, box2d_list, score_list):
#     ''' Write frustum pointnets results to KITTI format label files. '''
#     if result_dir is None: return
#     results = {} # map from idx to list of strings, each string is a line (without \n)
#     for i in range(len(score_list)):
#         idx = id_list[i]
#         output_str = type_list[i] + " -1 -1 -10 "
#         box2d = box2d_list[i]
#         output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
#         # h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
#         #     heading_cls_list[i], heading_res_list[i],
#         #     size_cls_list[i], size_res_list[i], rot_angle_list[i])
#         score = score_list[i]
#         # output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
#         if idx not in results: results[idx] = []
#         results[idx].append(output_str)
#
#     # Write TXT files
#     if not os.path.exists(result_dir): os.mkdir(result_dir)
#     output_dir = os.path.join(result_dir, 'data')
#     if not os.path.exists(output_dir): os.mkdir(output_dir)
#     for idx in results:
#         pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
#         fout = open(pred_filename, 'w')
#         for line in results[idx]:
#             fout.write(line+'\n')
#         fout.close()


def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()


# def test_from_rgb_detection(output_filename, result_dir=None):
#     ''' Test frustum pointents with 2D boxes from a RGB detector.
#     Write test results to KITTI format label files.
#     todo (rqi): support variable number of points.
#     '''
#     ps_list = []
#     segp_list = []
#     # center_list = []
#     # heading_cls_list = []
#     # heading_res_list = []
#     # size_cls_list = []
#     # size_res_list = []
#     # rot_angle_list = []
#     score_list = []
#     onehot_list = []
#
#     test_idxs = np.arange(0, len(TEST_DATASET))
#     print(len(TEST_DATASET))
#     batch_size = BATCH_SIZE
#     num_batches = int((len(TEST_DATASET)+batch_size-1)/batch_size)
#
#     batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))
#     batch_one_hot_to_feed = np.zeros((batch_size, 3))
#     sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
#     for batch_idx in range(num_batches):
#         print('batch idx: %d' % (batch_idx))
#         start_idx = batch_idx * batch_size
#         end_idx = min(len(TEST_DATASET), (batch_idx+1) * batch_size)
#         cur_batch_size = end_idx - start_idx
#
#         batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec = \
#             get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
#                 NUM_POINT, NUM_CHANNEL, from_rgb_detection=True)
#         batch_data_to_feed[0:cur_batch_size,...] = batch_data
#         batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec
#
#         # Run one batch inference
# 	# batch_output, batch_center_pred, \
#     #     batch_hclass_pred, batch_hres_pred, \
#     #     batch_sclass_pred, batch_sres_pred, batch_scores = \
#     #         inference(sess, ops, batch_data_to_feed,
#     #             batch_one_hot_to_feed, batch_size=batch_size)
# 	batch_output, \
#         batch_scores = \
#             inference(sess, ops, batch_data_to_feed,
#                 batch_one_hot_to_feed, batch_size=batch_size)
#
#         for i in range(cur_batch_size):
#             ps_list.append(batch_data[i,...])
#             segp_list.append(batch_output[i,...])
#             # center_list.append(batch_center_pred[i,:])
#             # heading_cls_list.append(batch_hclass_pred[i])
#             # heading_res_list.append(batch_hres_pred[i])
#             # size_cls_list.append(batch_sclass_pred[i])
#             # size_res_list.append(batch_sres_pred[i,:])
#             # rot_angle_list.append(batch_rot_angle[i])
#             #score_list.append(batch_scores[i])
#             score_list.append(batch_rgb_prob[i]) # 2D RGB detection score
#             onehot_list.append(batch_one_hot_vec[i])
#
#     if FLAGS.dump_result:
#         with open(output_filename, 'wp') as fp:
#             pickle.dump(ps_list, fp)
#             pickle.dump(segp_list, fp)
#             # pickle.dump(center_list, fp)
#             # pickle.dump(heading_cls_list, fp)
#             # pickle.dump(heading_res_list, fp)
#             # pickle.dump(size_cls_list, fp)
#             # pickle.dump(size_res_list, fp)
#             # pickle.dump(rot_angle_list, fp)
#             pickle.dump(score_list, fp)
#             pickle.dump(onehot_list, fp)
#
#     # Write detection results for KITTI evaluation
#     print('Number of point clouds: %d' % (len(ps_list)))
#     # write_detection_results(result_dir, TEST_DATASET.id_list,
#     #     TEST_DATASET.type_list, TEST_DATASET.box2d_list,
#     #     center_list, heading_cls_list, heading_res_list,
#     #     size_cls_list, size_res_list, rot_angle_list, score_list)
#     write_detection_results(result_dir, TEST_DATASET.id_list,
#         TEST_DATASET.type_list, score_list)
#     # Make sure for each frame (no matter if we have measurment for that frame),
#     # there is a TXT file
#     output_dir = os.path.join(result_dir, 'data')
#     if FLAGS.idx_path is not None:
#         to_fill_filename_list = [line.rstrip()+'.txt' \
#             for line in open(FLAGS.idx_path)]
#         fill_files(output_dir, to_fill_filename_list)

def test(output_filename, result_dir=None):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    # input_pc_list = []
    # gt_labels_list = []
    # pred_labels_list = []
    # center_list = []
    # heading_cls_list = []
    # heading_res_list = []
    # size_cls_list = []
    # size_res_list = []
    # rot_angle_list = []
    score_list = []

    sess, ops = get_session_and_ops()

    epsilon = 1e-12

    # box class-level metrics
    tp_sum = np.zeros(NUM_REAL_CLASSES)
    box_fn_sum = np.zeros(NUM_REAL_CLASSES)
    fp_sum = np.zeros(NUM_REAL_CLASSES)

    # box instance-level metrics
    iiou_sum = np.zeros(NUM_REAL_CLASSES)
    ire_sum = np.zeros(NUM_REAL_CLASSES)
    ipr_sum = np.zeros(NUM_REAL_CLASSES)
    i_sum = np.zeros(NUM_REAL_CLASSES)

    epoch_boxes_left_to_sample = []
    epoch_boxes_points_idxs_left_to_sample = []
    for i in np.arange(0, len(TEST_DATASET)):
        epoch_boxes_left_to_sample.append(i)
        epoch_boxes_points_idxs_left_to_sample.append(range(np.size(TEST_DATASET.pc_in_box_list[i], 0)))

    while len(epoch_boxes_left_to_sample) > 0:
        num_batches = int(np.ceil(len(epoch_boxes_left_to_sample) * 1.0 / BATCH_SIZE))
        new_epoch_boxes_left_to_sample = []
        new_epoch_boxes_points_idxs_left_to_sample = []
        print(num_batches)

        for batch_idx in range(num_batches):
            print('batch idx: %d' % (batch_idx))
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx+BATCH_SIZE, len(epoch_boxes_left_to_sample))

            batch_box_idxs = epoch_boxes_left_to_sample[start_idx:end_idx]
            batch_box_points_left_to_sample = epoch_boxes_points_idxs_left_to_sample[start_idx:end_idx]

            # batch_input_pc, batch_gt_labels, batch_center, \
            # batch_hclass, batch_hres, batch_sclass, batch_sres, \
            # batch_rot_angle, batch_one_hot_vec = \
            #     get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
            #         NUM_POINT, NUM_CHANNEL)
            batch_input_pc, batch_gt_labels, batch_rot_angle, batch_box_certainty, \
                new_batch_box_points_left_to_sample, batch_one_hot_vec = \
                get_batch(TEST_DATASET, BATCH_SIZE, batch_box_idxs, NUM_POINT, NUM_CHANNEL, NUM_REAL_CLASSES,
                          batch_box_points_left_to_sample)
            for i in range(len(batch_box_idxs)):
                if len(new_batch_box_points_left_to_sample[i]) > 0:
                    new_epoch_boxes_left_to_sample.append(batch_box_idxs[i])
                    new_epoch_boxes_points_idxs_left_to_sample.append(new_batch_box_points_left_to_sample[i])

        # pc_pred_labels, batch_center_pred, \
        #     batch_hclass_pred, batch_hres_pred, \
        #     batch_sclass_pred, batch_sres_pred, batch_scores = \
        #         inference(sess, ops, batch_input_pc,
        #                   batch_one_hot_vec, batch_size=batch_size)
            pc_pred_labels, batch_scores = \
                inference(sess, ops, batch_input_pc, batch_one_hot_vec, batch_box_certainty, batch_size=BATCH_SIZE)

            icare = batch_gt_labels != -1

            tps = np.sum(batch_gt_labels * pc_pred_labels * icare, 1)
            fns = np.sum(batch_gt_labels * (1 - pc_pred_labels * icare), 1)
            fps = np.sum((1- batch_gt_labels) * pc_pred_labels * icare, 1)

            iiou = tps.astype(np.float) / (tps + fns + fps + epsilon)
            ipr = tps.astype(np.float) / (tps + fps + epsilon)
            ire = tps.astype(np.float) / (tps + fns + epsilon)

            for i in range(BATCH_SIZE):
                point_percentage = np.sum(icare[i, :]) * 1.0 \
                                   / len(TEST_DATASET.pc_in_box_label_list[batch_box_idxs[i]])
                iiou_sum[batch_one_hot_vec[i, :]] += point_percentage * iiou[i]
                ire_sum[batch_one_hot_vec[i, :]] += point_percentage * ire[i]
                ipr_sum[batch_one_hot_vec[i, :]] += point_percentage * ipr[i]
                i_sum[batch_one_hot_vec[i, :]] += point_percentage

                tp_sum[batch_one_hot_vec[i, :]] += tps[i]
                box_fn_sum[batch_one_hot_vec[i, :]] += fns[i]
                fp_sum[batch_one_hot_vec[i, :]] += fps[i]

            for i in range(pc_pred_labels.shape[0]):
                # input_pc_list.append(batch_input_pc[i, ...])
                # gt_labels_list.append(batch_gt_labels[i, ...])
                # pred_labels_list.append(pc_pred_labels[i, ...])
                # center_list.append(batch_center_pred[i,:])
                # heading_cls_list.append(batch_hclass_pred[i])
                # heading_res_list.append(batch_hres_pred[i])
                # size_cls_list.append(batch_sclass_pred[i])
                # size_res_list.append(batch_sres_pred[i,:])
                # rot_angle_list.append(batch_rot_angle[i])
                score_list.append(batch_scores[i])
        epoch_boxes_points_idxs_left_to_sample = new_epoch_boxes_points_idxs_left_to_sample
        epoch_boxes_left_to_sample = new_epoch_boxes_left_to_sample

    box_ious = tp_sum.astype(np.float)/(tp_sum + box_fn_sum + fp_sum + epsilon)
    box_prs = tp_sum.astype(np.float)/(tp_sum + fp_sum + epsilon)
    box_res = tp_sum.astype(np.float)/(tp_sum + box_fn_sum + epsilon)

    iious = iiou_sum.astype(np.float) / (i_sum + epsilon)
    iprs = ipr_sum.astype(np.float) / (i_sum + epsilon)
    ires = ire_sum.astype(np.float) / (i_sum + epsilon)

    print("Segmentation results (box level):")
    print(REAL_CLASSES)
    print("IOU:")
    print(box_ious)
    print("Precision: (same as image level)")
    print(box_prs)
    print("Recall")
    print(box_res)
    print("instance IOU:")
    print(iious)
    print("instance Precision:")
    print(iprs)
    print("instance Recall")
    print(ires)

    # image class-level metrics
    image_fn_sum = box_fn_sum + TEST_DATASET.image_boxes_fn
    image_ious = tp_sum.astype(np.float)/(tp_sum + image_fn_sum + fp_sum + epsilon)
    image_prs = box_prs
    image_res = tp_sum.astype(np.float)/(tp_sum + image_fn_sum + epsilon)
    print("Segmentation results (box level):")
    print(REAL_CLASSES)
    print("IOU:")
    print(image_ious)
    print("Precision: (same as box level)")
    print(image_prs)
    print("Recall")
    print(image_res)

    if FLAGS.dump_result:
        with open(output_filename, 'wp') as fp:
            # pickle.dump(input_pc_list, fp)
            # pickle.dump(gt_labels_list, fp)
            # pickle.dump(pred_labels_list, fp)
            # pickle.dump(center_list, fp)
            # pickle.dump(heading_cls_list, fp)
            # pickle.dump(heading_res_list, fp)
            # pickle.dump(size_cls_list, fp)
            # pickle.dump(size_res_list, fp)
            # pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)

    # Write detection results for KITTI evaluation
    # write_detection_results(result_dir, TEST_DATASET.id_list,
    #     TEST_DATASET.type_list, TEST_DATASET.box2d_list, center_list,
    #     heading_cls_list, heading_res_list,
    #     size_cls_list, size_res_list, rot_angle_list, score_list)
    # write_detection_results(result_dir, TEST_DATASET.id_list,
    #     TEST_DATASET.type_list, TEST_DATASET.box2d_list, score_list, box_ious, pr, re)


if __name__=='__main__':
    # if FLAGS.from_rgb_detection:
    #     test_from_rgb_detection(FLAGS.output+'.pickle', FLAGS.output)
    # else:
    #     test(FLAGS.output+'.pickle', FLAGS.output)

    global FLAGS, BATCH_SIZE, NUM_POINT, GPU_INDEX, NUM_CHANNEL, TEST_DATASET, MODEL_PATH, MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
    parser.add_argument('--model_path', default='log/model.ckpt',
                        help='model checkpoint file path [default: log/model.ckpt]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
    parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
    parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
    parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
    parser.add_argument('--idx_path', default=None,
                        help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
    parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
    parser.add_argument('--with_intensity', action='store_true', help='Use Intensity for training')
    parser.add_argument('--with_colors', action='store_true', help='Use Colors for training')
    parser.add_argument('--with_depth_confidences', action='store_true', help='Use depth completion confidences')
    parser.add_argument('--from_guided_depth_completion', action='store_true',
                        help='Use point cloud from depth completion')
    parser.add_argument('--from_unguided_depth_completion', action='store_true',
                        help='Use point cloud from unguided depth completion')
    parser.add_argument('--from_depth_prediction', action='store_true', help='Use point cloud from depth prediction')
    parser.add_argument('--restore_model_path', default=None,
                        help='Restore model path e.g. log/model.ckpt [default: None]')
    parser.add_argument('--dont_input_box_probabilities', action='store_true',
                        help='Use box probabilities as net inputs')
    parser.add_argument('--avoid_point_duplicates', action='store_true',
                        help='Try to avoid point duplicates when sampling')
    FLAGS = parser.parse_args()

    # Set training configurations
    BATCH_SIZE = FLAGS.batch_size
    MODEL_PATH = FLAGS.model_path
    GPU_INDEX = FLAGS.gpu
    NUM_POINT = FLAGS.num_point
    MODEL = importlib.import_module(FLAGS.model)
    NUM_CHANNEL = 4

    # Load Frustum Datasets.
    print('--- Loading Testing Dataset ---')
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val', classes=REAL_CLASSES,
                                           random_flip=False, random_shift=False,
                                           rotate_to_center=True, box_class_one_hot=True, from_rgb_detection=True,
                                           with_color=FLAGS.with_colors, with_intensity=FLAGS.with_intensity,
                                           with_depth_confidences=FLAGS.with_depth_confidences,
                                           from_guided_depth_completion=FLAGS.from_guided_depth_completion,
                                           from_unguided_depth_completion=FLAGS.from_unguided_depth_completion,
                                           from_depth_prediction=FLAGS.from_depth_prediction,
                                           segment_all_points=True, avoid_duplicates=FLAGS.avoid_point_duplicates)
    test(FLAGS.output + '.pickle', FLAGS.output)