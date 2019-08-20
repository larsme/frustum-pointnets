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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
from train_util import get_batch
from mkdir_p import mkdir_p


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--with_intensity', action='store_true', help='Use Intensity for training')
parser.add_argument('--with_colors', action='store_true', help='Use Colors for training')
parser.add_argument('--with_depth_confidences', action='store_true', help='Use depth completion confidences')
parser.add_argument('--from_guided_depth_completion', action='store_true', help='Use point cloud from depth completion')
parser.add_argument('--from_unguided_depth_completion', action='store_true',
                    help='Use point cloud from unguided depth completion')
parser.add_argument('--from_depth_prediction', action='store_true', help='Use point cloud from depth prediction')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--dont_input_box_probabilities', action='store_true', help='Use box probabilities as net inputs')
parser.add_argument('--avoid_point_duplicates', action='store_true', help='Try to avoid point duplicates when sampling')
FLAGS = parser.parse_args()


# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
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



#Load Frustum Datasets. Use default data paths
print('--- Loading Training Dataset ---')
TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train', classes=REAL_CLASSES,
                                        random_flip=True, random_shift=False,
                                        rotate_to_center=True, box_class_one_hot=True, from_rgb_detection=True,
                                        with_color=FLAGS.with_colors, with_intensity=FLAGS.with_intensity,
                                        with_depth_confidences=FLAGS.with_depth_confidences,
                                        from_guided_depth_completion=FLAGS.from_guided_depth_completion,
                                        from_unguided_depth_completion=FLAGS.from_unguided_depth_completion,
                                        from_depth_prediction=FLAGS.from_depth_prediction,
                                        segment_all_points=False, avoid_duplicates=FLAGS.avoid_point_duplicates)
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

print('--- Loading Model ---')
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.isdir(LOG_DIR):
    mkdir_p(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')



BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99




def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():


    ''' Main function for training and simple evaluation. '''
    with tf.Graph().as_default():
        epsilon = tf.constant(1e-12, tf.float32, name='epsilon')
        with tf.device('/gpu:'+str(GPU_INDEX)):

            with tf.variable_scope("learn_preperation"):
                is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')

                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to increment the 'batch' parameter
                # for you every time it trains.
                batch = tf.get_variable('batch', [],
                    initializer=tf.constant_initializer(0), trainable=False)
                bn_decay = get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

            with tf.name_scope("batch_input"):
                batch_input_pc, batch_box_certainty, batch_gt_labels, batch_one_hot_vec = \
                    MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_CHANNEL, NUM_REAL_CLASSES)

                if FLAGS.dont_input_box_probabilities:
                    batch_box_label_prob = tf.to_float(batch_one_hot_vec, 'batch_one_hot_vec')
                else:
                    batch_box_label_prob = tf.multiply(tf.to_float(batch_one_hot_vec),
                                                       tf.tile(tf.expand_dims(batch_box_certainty, axis=1),
                                                               [1, NUM_REAL_CLASSES]),
                                                       'batch_box_label_prob')

                icare = tf.cast(tf.not_equal(batch_gt_labels, -1), tf.int32, 'icare')
                num_labeled_points = tf.to_float(tf.reduce_sum(icare), 'num_labeled_points') + epsilon

            # Get model and losses
            # end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
            #     is_training_pl, bn_decay=bn_decay)

            # logits for element or no element (not just 1 prob)
            pc_pred_logits = MODEL.get_model(batch_input_pc, batch_box_label_prob, is_training_pl, bn_decay=bn_decay)

            with tf.name_scope("batch_metrics"):
                pc_pred_labels = tf.argmax(pc_pred_logits, axis=2, output_type=tf.int32, name='pc_pred_labels')
                tps = tf.to_float(tf.reduce_sum(batch_gt_labels * pc_pred_labels * icare, axis=0), 'tps')
                fns = tf.to_float(tf.reduce_sum(batch_gt_labels * (1 - pc_pred_labels) * icare, axis=0), 'fns')
                fps = tf.to_float(tf.reduce_sum((1 - batch_gt_labels) * pc_pred_labels * icare, axis=0), 'fps')
                tns = tf.to_float(tf.reduce_sum((1 - batch_gt_labels) * (1 - pc_pred_labels) * icare, axis=0), 'tns')

                iiou = tps / (tps + fns + fps + epsilon)
                ipr = tps / (tps + fps + epsilon)
                ire = tps / (tps + fns + epsilon)

                tf.summary.scalar('iIOU', tf.reduce_mean(iiou))
                tf.summary.scalar('iPrecision', tf.reduce_mean(ipr))
                tf.summary.scalar('iRecall', tf.reduce_mean(ire))

                # Write summaries of bounding box IoU and segmentation accuracies
                # iou2ds, iou3ds = tf.py_func(provider.compute_box3d_iou, [\
                #     end_points['center'], \
                #     end_points['heading_scores'], end_points['heading_residuals'], \
                #     end_points['size_scores'], end_points['size_residuals'], \
                #     centers_pl, \
                #     heading_class_label_pl, heading_residual_label_pl, \
                #     size_class_label_pl, size_residual_label_pl], \
                #     [tf.float32, tf.float32])
                # end_points['iou2ds'] = iou2ds
                # end_points['iou3ds'] = iou3ds
                # tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
                # tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))

                # correct = tf.equal(tf.argmax(end_points['mask_logits'], 2),
                #     tf.to_int64(labels_pl))
                # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / \
                #     float(BATCH_SIZE*NUM_POINT)
                # tf.summary.scalar('segmentation accuracy', accuracy)
                tf.summary.scalar('segmentation accuracy',
                                  tf.reduce_sum(tps+tns) / num_labeled_points)

            with tf.variable_scope("learning"):
                # loss = MODEL.get_loss(labels_pl, centers_pl,
                #     heading_class_label_pl, heading_residual_label_pl,
                #     size_class_label_pl, size_residual_label_pl, end_points)

                # 3D Segmentation loss
                loss = MODEL.get_loss(batch_gt_labels, pc_pred_logits, icare)
                tf.summary.scalar('loss', loss)
                # losses = tf.get_collection('losses')
                # total_loss = tf.add_n(losses, name='total_loss')
                # tf.summary.scalar('total_loss', total_loss)

                # Get training operator
                learning_rate = get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
        class_writers = []
        for class_idx in range(NUM_REAL_CLASSES):
            class_writers.append(tf.summary.FileWriter(os.path.join(LOG_DIR, REAL_CLASSES[class_idx]), sess.graph))
        class_writers.append(tf.summary.FileWriter(os.path.join(LOG_DIR, 'Mean Class'), sess.graph))
        class_writers.append(tf.summary.FileWriter(os.path.join(LOG_DIR, 'Class Independent'), sess.graph))

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

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
        #        # 'centers_pred': end_points['center'],
        #        'loss': loss,
        #        'train_op': train_op,
        #        'merged': merged,
        #        'step': batch,
        #        'end_points': end_points}
        ops = {'batch_input_pc': batch_input_pc,
               'batch_box_certainty': batch_box_certainty,
               'batch_gt_labels': batch_gt_labels,
               'batch_one_hot_vec': batch_one_hot_vec,
               'is_training_pl': is_training_pl,
               'pc_pred_logits': pc_pred_logits,
               # 'centers_pred': end_points['center'],
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        # eval_one_epoch(sess, ops, test_writer, class_writers)
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer, class_writers)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    ''' Training for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops
    '''
    is_training = True
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d TRAINING ----' % EPOCH_CNT)

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = int(np.ceil(len(TRAIN_DATASET)/BATCH_SIZE))

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    # iou2ds_sum = 0
    # iou3ds_sum = 0
    # iou3d_correct_cnt = 0

    # Training with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx+BATCH_SIZE, len(train_idxs))
        batch_idxs = train_idxs[start_idx:end_idx]

        # batch_data, batch_label, batch_center, \
        # batch_hclass, batch_hres, \
        # batch_sclass, batch_sres, \
        # batch_rot_angle, batch_one_hot_vec = \
        #     get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx,
        #         NUM_POINT, NUM_CHANNEL)
        batch_input_pc, batch_gt_labels, batch_rot_angle, batch_box_certainty, _, batch_one_hot_vec = \
            get_batch(TRAIN_DATASET, BATCH_SIZE, batch_idxs, NUM_POINT, NUM_CHANNEL, NUM_REAL_CLASSES)

        # feed_dict = {ops['pointclouds_pl']: batch_data,
        #              ops['one_hot_vec_pl']: batch_one_hot_vec,
        #              ops['labels_pl']: batch_label,
        #              ops['centers_pl']: batch_center,
        #              ops['heading_class_label_pl']: batch_hclass,
        #              ops['heading_residual_label_pl']: batch_hres,
        #              ops['size_class_label_pl']: batch_sclass,
        #              ops['size_residual_label_pl']: batch_sres,
        #              ops['is_training_pl']: is_training, }
        feed_dict = {ops['batch_input_pc']: batch_input_pc,
                     ops['batch_box_certainty']: batch_box_certainty,
                     ops['batch_gt_labels']: batch_gt_labels,
                     ops['batch_one_hot_vec']: batch_one_hot_vec,
                     ops['is_training_pl']: is_training}

        # summary, step, _, loss, pc_pred_logits, centers_pred_val, \
        # iou2ds, iou3ds = \
        #     sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
        #         ops['logits'], ops['centers_pred'],
        #         ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
        #         feed_dict=feed_dict)
        summary, step, _, loss, pc_pred_logits = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pc_pred_logits']],
                     feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

        pc_pred_labels_val = np.argmax(pc_pred_logits, 2)
        correct = np.sum(pc_pred_labels_val == batch_gt_labels)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss
        # iou2ds_sum += np.sum(iou2ds)
        # iou3ds_sum += np.sum(iou3ds)
        # iou3d_correct_cnt += np.sum(iou3ds>=0.7)

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / num_batches))
            log_string('segmentation accuracy: %f' % \
                (total_correct / float(total_seen)))
            # log_string('box IoU (ground/3D): %f / %f' % \
            #     (iou2ds_sum / float(BATCH_SIZE*10), iou3ds_sum / float(BATCH_SIZE*10)))
            # log_string('box estimation accuracy (IoU=0.7): %f' % \
            #     (float(iou3d_correct_cnt)/float(BATCH_SIZE*10)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            # iou2ds_sum = 0
            # iou3ds_sum = 0
            # iou3d_correct_cnt = 0


def eval_one_epoch(sess, ops, test_writer, class_writers):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % EPOCH_CNT)

    # To collect statistics
    total_correct = 0
    total_seen = 0
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

    epoch_boxes_left_to_sample = []
    epoch_boxes_points_idxs_left_to_sample = []
    for i in np.arange(0, len(TEST_DATASET)):
        epoch_boxes_left_to_sample.append(i)
        epoch_boxes_points_idxs_left_to_sample.append(range(np.size(TEST_DATASET.pc_in_box_list[i], 0)))

    while len(epoch_boxes_left_to_sample) > 0:
        new_epoch_boxes_left_to_sample = []
        new_epoch_boxes_points_idxs_left_to_sample = []

        num_batches = int(np.ceil(len(epoch_boxes_left_to_sample) * 1.0 / BATCH_SIZE))
        new_epoch_boxes_left_to_sample = []
        new_epoch_boxes_points_idxs_left_to_sample = []
        print(num_batches)

        # Simple evaluation with batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx+BATCH_SIZE, len(epoch_boxes_left_to_sample))

            batch_box_idxs = epoch_boxes_left_to_sample[start_idx:end_idx]
            batch_box_points_left_to_sample = epoch_boxes_points_idxs_left_to_sample[start_idx:end_idx]

            # batch_data, batch_label, batch_center, \
            # batch_hclass, batch_hres, \
            # batch_sclass, batch_sres, \
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

            # feed_dict = {ops['pointclouds_pl']: batch_data,
            #              ops['one_hot_vec_pl']: batch_one_hot_vec,
            #              ops['labels_pl']: batch_label,
            #              ops['centers_pl']: batch_center,
            #              ops['heading_class_label_pl']: batch_hclass,
            #              ops['heading_residual_label_pl']: batch_hres,
            #              ops['size_class_label_pl']: batch_sclass,
            #              ops['size_residual_label_pl']: batch_sres,
            #              ops['is_training_pl']: is_training}
            feed_dict = {ops['batch_input_pc']: batch_input_pc,
                         ops['batch_box_certainty']: batch_box_certainty,
                         ops['batch_gt_labels']: batch_gt_labels,
                         ops['batch_one_hot_vec']: batch_one_hot_vec,
                         ops['is_training_pl']: is_training}

            # summary, step, loss_val, logits_val, iou2ds, iou3ds = \
            #     sess.run([ops['merged'], ops['step'],
            #         ops['loss'], ops['logits'],
            #         ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
            #         feed_dict=feed_dict)
            summary, step, _, loss, pc_pred_logits = \
                sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pc_pred_logits']],
                         feed_dict=feed_dict)

            test_writer.add_summary(summary, step)

            pc_pred_labels = np.argmax(pc_pred_logits, 2)
            loss_sum += loss
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum(batch_gt_labels == l)
                total_correct_class[l] += (np.sum((pc_pred_labels == l) & (batch_gt_labels == l)))

            icare = batch_gt_labels != -1

            tps = np.sum(batch_gt_labels * pc_pred_labels * icare, 1)
            fns = np.sum(batch_gt_labels * (1 - pc_pred_labels) * icare, 1)
            fps = np.sum((1 - batch_gt_labels) * pc_pred_labels * icare, 1)
            tns = np.sum((1 - batch_gt_labels) * (1 - pc_pred_labels) * icare, 1)

            iiou = tps.astype(np.float) / (tps + fns + fps + epsilon)
            ipr = tps.astype(np.float) / (tps + fps + epsilon)
            ire = tps.astype(np.float) / (tps + fns + epsilon)

            for i in range(end_idx - start_idx):
                point_percentage = np.sum(icare[i, :]) * 1.0 \
                                   / len(TEST_DATASET.pc_in_box_label_list[batch_box_idxs[i]])
                iiou_sum[batch_one_hot_vec[i, :]] += point_percentage * iiou[i]
                ire_sum[batch_one_hot_vec[i, :]] += point_percentage * ire[i]
                ipr_sum[batch_one_hot_vec[i, :]] += point_percentage * ipr[i]
                i_sum[batch_one_hot_vec[i, :]] += point_percentage

                tp_sum[batch_one_hot_vec[i, :]] += tps[i]
                box_fn_sum[batch_one_hot_vec[i, :]] += fns[i]
                fp_sum[batch_one_hot_vec[i, :]] += fps[i]
                tn_sum[batch_one_hot_vec[i, :]] += tns[i]

        epoch_boxes_points_idxs_left_to_sample = new_epoch_boxes_points_idxs_left_to_sample
        epoch_boxes_left_to_sample = new_epoch_boxes_left_to_sample

    log_string('eval mean loss: %f' % (loss_sum / np.sum(i_sum)))
    log_string('eval segmentation accuracy: %f'% (float(np.sum(tp_sum+tn_sum))
               /(float(np.sum(tp_sum + box_fn_sum + fp_sum + tn_sum)) + epsilon)))
    log_string('eval segmentation avg class acc: %f'% np.mean((tp_sum+tn_sum).astype(np.float)
                                                              / (tp_sum + box_fn_sum + fp_sum + tn_sum + epsilon)))
    # log_string('eval box IoU (ground/3D): %f / %f' % \
    #     (iou2ds_sum / float(num_batches*BATCH_SIZE), iou3ds_sum / \
    #         float(num_batches*BATCH_SIZE)))
    # log_string('eval box estimation accuracy (IoU=0.7): %f' % \
    #     (float(iou3d_correct_cnt)/float(num_batches*BATCH_SIZE)))

    box_ious = tp_sum.astype(np.float)/(tp_sum + box_fn_sum + fp_sum + epsilon)
    box_prs = tp_sum.astype(np.float)/(tp_sum + fp_sum + epsilon)
    box_res = tp_sum.astype(np.float)/(tp_sum + box_fn_sum + epsilon)

    box_any_ious = np.sum(tp_sum).astype(np.float)/(np.sum(tp_sum + box_fn_sum + fp_sum) + epsilon)
    box_any_prs = np.sum(tp_sum).astype(np.float)/(np.sum(tp_sum + fp_sum) + epsilon)
    box_any_res = np.sum(tp_sum).astype(np.float)/(np.sum(tp_sum + box_fn_sum) + epsilon)

    iious = iiou_sum.astype(np.float) / (i_sum + epsilon)
    iprs = ipr_sum.astype(np.float) / (i_sum + epsilon)
    ires = ire_sum.astype(np.float) / (i_sum + epsilon)

    iious_any = np.sum(iiou_sum).astype(np.float) / (np.sum(i_sum) + epsilon)
    iprs_any = np.sum(ipr_sum).astype(np.float) / (np.sum(i_sum) + epsilon)
    ires_any = np.sum(ire_sum).astype(np.float) / (np.sum(i_sum) + epsilon)

    # image class-level metrics
    image_fn_sum = box_fn_sum + TEST_DATASET.image_boxes_fn
    image_ious = tp_sum.astype(np.float)/(tp_sum + image_fn_sum + fp_sum + epsilon)
    image_prs = box_prs
    image_res = tp_sum.astype(np.float)/(tp_sum + image_fn_sum + epsilon)

    image_any_ious = np.sum(tp_sum).astype(np.float)/(np.sum(tp_sum + image_fn_sum + fp_sum) + epsilon)
    image_any_prs = box_any_prs
    image_any_res = np.sum(tp_sum).astype(np.float)/(np.sum(tp_sum + image_fn_sum) + epsilon)

    for class_idx in range(NUM_REAL_CLASSES):
        summary = tf.Summary()
        summary.value.add(tag='Box_IOU', simple_value=box_ious[class_idx])
        summary.value.add(tag='Box_Precision', simple_value=box_prs[class_idx])
        summary.value.add(tag='Box_Recall', simple_value=box_res[class_idx])
        summary.value.add(tag='Box_iIOU', simple_value=iious[class_idx])
        summary.value.add(tag='Box_iPrecision', simple_value=iprs[class_idx])
        summary.value.add(tag='Box_iRecall', simple_value=ires[class_idx])
        summary.value.add(tag='Image_IOU', simple_value=image_ious[class_idx])
        summary.value.add(tag='Image_Precision', simple_value=image_prs[class_idx])
        summary.value.add(tag='Image_Recall', simple_value=image_res[class_idx])
        class_writers[class_idx].add_summary(summary, EPOCH_CNT)

    summary = tf.Summary()
    summary.value.add(tag='Box_IOU', simple_value=np.mean(box_ious))
    summary.value.add(tag='Box_Precision', simple_value=np.mean(box_prs))
    summary.value.add(tag='Box_Recall', simple_value=np.mean(box_res))
    summary.value.add(tag='Box_iIOU', simple_value=np.mean(iious))
    summary.value.add(tag='Box_iPrecision', simple_value=np.mean(iprs))
    summary.value.add(tag='Box_iRecall', simple_value=np.mean(ires))
    summary.value.add(tag='Image_IOU', simple_value=np.mean(image_ious))
    summary.value.add(tag='Image_Precision', simple_value=np.mean(image_prs))
    summary.value.add(tag='Image_Recall', simple_value=np.mean(image_res))
    class_writers[NUM_REAL_CLASSES].add_summary(summary, EPOCH_CNT)

    summary = tf.Summary()
    summary.value.add(tag='Box_IOU', simple_value=np.mean(box_any_ious))
    summary.value.add(tag='Box_Precision', simple_value=np.mean(box_any_prs))
    summary.value.add(tag='Box_Recall', simple_value=np.mean(box_any_res))
    summary.value.add(tag='Box_iIOU', simple_value=np.mean(iious_any))
    summary.value.add(tag='Box_iPrecision', simple_value=np.mean(iprs_any))
    summary.value.add(tag='Box_iRecall', simple_value=np.mean(ires_any))
    summary.value.add(tag='Image_IOU', simple_value=np.mean(image_any_ious))
    summary.value.add(tag='Image_Precision', simple_value=np.mean(image_any_prs))
    summary.value.add(tag='Image_Recall', simple_value=np.mean(image_any_res))
    class_writers[NUM_REAL_CLASSES+1].add_summary(summary, EPOCH_CNT)

    EPOCH_CNT += 1


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
