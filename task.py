#encoding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict
import model
import train_operation as op
import time

MAX_STEPS = 10000000
LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 1
TRAIN_FILE = "train_small.csv"
VALIDATION_FILE = "validate_small.csv"
COARSE_DIR = "coarse_small"
REFINE_DIR = "refine_small"
PREDICT_REFINE_DIR = "data/predict_refine_small"
PREDICT_DIR = "data/predict_small"
VALIDATION_DIR = "data/validation_small"

REFINE_TRAIN = False
FINE_TUNE = True

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(BATCH_SIZE)
        val_dataset = DataSet(BATCH_SIZE)
        images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
        val_images, val_depths, val_invalid_depths = val_dataset.csv_inputs(VALIDATION_FILE)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        val_keep_conv = tf.placeholder(tf.float32)
        val_keep_hidden = tf.placeholder(tf.float32)
        if REFINE_TRAIN:
            print("refine train.")
            coarse = model.inference(images, keep_conv, trainable=False)
            logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)
        else:
            print("coarse train.")
            logits = model.inference(images, keep_conv, keep_hidden)
            val_logits = model.inference(val_images, reuse=True, trainable=False)
        loss = model.loss(logits, depths, invalid_depths)
        val_loss = model.loss(val_logits, val_depths, val_invalid_depths)
        train_op = op.train(loss, global_step, BATCH_SIZE)
        val_op = op.train(val_loss, global_step, BATCH_SIZE)
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init_op)    

        # parameters
        coarse_params = {}
        refine_params = {}
        if REFINE_TRAIN:
            for variable in tf.all_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                print("parameter: %s" %(variable_name))
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        else:
            for variable in tf.trainable_variables():
                variable_name = variable.name
                print("parameter: %s" %(variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        # define saver
        print(coarse_params)
        saver_coarse = tf.train.Saver(coarse_params)
        if REFINE_TRAIN:
            saver_refine = tf.train.Saver(refine_params)
        # fine tune
        if FINE_TUNE:
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
            if REFINE_TRAIN:
                refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
                if refine_ckpt and refine_ckpt.model_checkpoint_path:
                    print("Pretrained refine Model Loading.")
                    saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                    print("Pretrained refine Model Restored.")
                else:
                    print("No Pretrained refine Model.")

        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(MAX_STEPS):
            index = 0
            for i in range(1000):
                _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                _, val_loss_value, val_logits_val, val_images_val = sess.run([val_op, val_loss, val_logits, val_images])
                if index % 10 == 0:
                    # print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))
                    print("%s: %d[epoch]: %d[iteration]: train loss %f validation loss %f" % (datetime.now(), step, index, loss_value, val_loss_value))
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if index % 500 == 0:
                    if REFINE_TRAIN:
                        output_predict(logits_val, images_val, "%s_%05d_%05d" % (PREDICT_REFINE_DIR, step, i))
                    else:
                        output_predict(logits_val, images_val, "%s_%05d_%05d" % (PREDICT_DIR, step, i))
                    output_predict(val_logits_val, val_images_val, "%s_%05d_%05d" % (VALIDATION_DIR, step, i))    
                index += 1

            if step % 5 == 0 or (step * 1) == MAX_STEPS:
                if REFINE_TRAIN:
                    refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path, global_step=step)
                else:
                    coarse_checkpoint_path = COARSE_DIR + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path, global_step=step)
        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()
