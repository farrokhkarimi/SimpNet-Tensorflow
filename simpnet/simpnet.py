# ========================================
# [] File Name : simpnet.py
#
# [] Creation Date : Aug 2018
#
# [] Author : Ali Gholami
#
# ========================================

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import time
from itertools import chain
from cnn_util import conv_bn_sc_relu, saf_pool
from cnn_config import *
from utils import *

class SimpNet(object):

    def __init__(self):

        # Dropout rate
        self.keep_prob = 0.7

        # Learning rate
        self.learning_rate = 0.001

        # Setup the data path (folder should contain train and test folders inside itself)
        self.data_path = './images/images'
        self.main_csv = 'Data_Entry_2017.csv'
        self.train_val_csv = shuffle_csv('train_val_list.csv')
        self.test_csv = shuffle_csv('test_list.csv')

        # Number of images in each batch
        self.batch_size = 8

        # Number of classes
        self.n_classes = 14

        # Global step (times the graph seen the data)
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Shows the overall graph status (Trainig vs Testing)
        self.training = True

        # Which steps show the loss in each epoch
        self.skip_steps = 500

        self.n_test = 25596

        self.train_list = pd.read_csv(self.main_csv)

        # Initial data preprocessing
        self.preprocess()

    def preprocess(self):
        # Convert to one hot
        self.train_list['Finding Labels'] = self.train_list['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
        self.all_labels = np.unique(list(chain(*self.train_list['Finding Labels'].map(lambda x: x.split('|')).tolist())))
        self.all_labels = [x for x in self.all_labels if len(x)>0]
        for c_label in self.all_labels:
            if len(c_label)>1: # leave out empty labels
                self.train_list[c_label] = self.train_list['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

    def get_data(self):

        with tf.name_scope('data'):
            
            train_data_generator = lambda: self.nih_data_generator(images_path=self.data_path, from_target=self.train_val_csv)
            test_data_generator = lambda: self.nih_data_generator(images_path=self.data_path, from_target=self.test_csv)

            train_data = tf.data.Dataset.from_generator(
            generator=train_data_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]))
            ).batch(self.batch_size).prefetch(2)
            
            test_data = tf.data.Dataset.from_generator(
            generator=test_data_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]))
            ).batch(self.batch_size).prefetch(2)

            iterator = train_data.make_initializable_iterator()

            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, [-1, CNN_INPUT_HEIGHT, CNN_INPUT_WIDTH, CNN_INPUT_CHANNELS])

            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)

    def nih_data_generator(self, images_path, from_target):
        with open(from_target, 'r') as f:
            for image in f.readlines():
                image = image.strip()
                img = os.path.join(images_path, image)
                a = cv2.imread(img)
                if a is None:
                    print("Unable to read image", img)
                    continue
                
                a = cv2.resize(a, (224, 224))
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                yield (np.array(a.flatten()), self.train_list.loc[self.train_list[self.train_list["Image Index"] == image].index.item(), self.all_labels].as_matrix())

    def build_network_graph(self):

        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def inference(self):

        conv1 = conv_bn_sc_relu(
            inputs=self.img,
            filters=CONV1_NUM_FILTERS,
            k_size=CONV1_FILTER_SIZE,
            stride=2,
            padding='SAME',
            scope_name='conv_1',
            keep_prob=self.keep_prob
        )
    
        conv2 = conv_bn_sc_relu(
            inputs=conv1,
            filters=CONV2_NUM_FILTERS,
            k_size=CONV2_FILTER_SIZE,
            stride=2,
            padding='SAME',
            scope_name='conv_2',
            keep_prob=self.keep_prob
        )

        conv3 = conv_bn_sc_relu(
            inputs=conv2,
            filters=CONV3_NUM_FILTERS,
            k_size=CONV3_FILTER_SIZE,
            stride=2,
            padding='SAME',
            scope_name='conv_3',
            keep_prob=self.keep_prob
        )

        conv4 = conv_bn_sc_relu(
            inputs=conv3,
            filters=CONV4_NUM_FILTERS,
            k_size=CONV4_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_4',
            keep_prob=self.keep_prob
        )

        conv5 = conv_bn_sc_relu(
            inputs=conv4,
            filters=CONV5_NUM_FILTERS,
            k_size=CONV5_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_5',
            keep_prob=self.keep_prob
        )

        pool1 = saf_pool(
            inputs=conv5,
            k_size=MAXPOOL1_SIZE,
            stride=2,
            padding='VALID',
            scope_name='saf_pool_1'
        )

        conv6 = conv_bn_sc_relu(
            inputs=pool1,
            filters=CONV6_NUM_FILTERS,
            k_size=CONV6_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_6',
            keep_prob=self.keep_prob
        )

        conv7 = conv_bn_sc_relu(
            inputs=conv6,
            filters=CONV7_NUM_FILTERS,
            k_size=CONV7_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_7',
            keep_prob=self.keep_prob
        )

        conv8 = conv_bn_sc_relu(
            inputs=conv7,
            filters=CONV8_NUM_FILTERS,
            k_size=CONV8_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_8',
            keep_prob=self.keep_prob
        )

        conv9 = conv_bn_sc_relu(
            inputs=conv8,
            filters=CONV9_NUM_FILTERS,
            k_size=CONV9_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_9',
            keep_prob=self.keep_prob
        )

        conv10 = conv_bn_sc_relu(
            inputs=conv9,
            filters=CONV10_NUM_FILTERS,
            k_size=CONV10_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_10',
            keep_prob=self.keep_prob
        )

        pool2 = saf_pool(
            inputs=conv10,
            k_size=MAXPOOL2_SIZE,
            stride=2,
            padding='VALID',
            scope_name='saf_pool_2'
        )

        conv11 = conv_bn_sc_relu(
            inputs=pool2,
            filters=CONV11_NUM_FILTERS,
            k_size=CONV11_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_11',
            keep_prob=self.keep_prob
        )

        conv12 = conv_bn_sc_relu(
            inputs=conv11,
            filters=CONV11_NUM_FILTERS,
            k_size=CONV11_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_12',
            keep_prob=self.keep_prob
        )

        conv13 = conv_bn_sc_relu(
            inputs=conv12,
            filters=CONV13_NUM_FILTERS,
            k_size=CONV13_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_13',
            keep_prob=self.keep_prob
        )

        global_pool = saf_pool(
            inputs=conv13,
            k_size=MAXPOOL3_SIZE,
            stride=2,
            padding='VALID',
            scope_name='global_saf_pool'
        )

        flattened = tf.layers.flatten(
            inputs=global_pool,
            name='flatten_input'
        )

        # Softmax is applied in the loss function when softmax_cross_entropy_with_logits is called
        self.logits = tf.layers.dense(
            inputs=flattened,
            units=self.n_classes,
            name='fc_final'
        )


    def loss(self):

        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)

            # Loss is mean of error on all dimensions
            self.loss_val = tf.reduce_mean(entropy, name='loss')


    def optimize(self):

        with tf.name_scope('optimizer'):
            self.opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                name='Adam'
            ).minimize(self.loss_val, global_step=self.gstep)

    def summary(self):

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss_val)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('loss histogram', self.loss_val)
            self.summary_op = tf.summary.merge_all()

    def eval(self):

        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)

            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))

            # Summation of all probabilities of all correct predictions
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    
    def train_network_one_epoch(self, sess, init, saver, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        print("[INFO]: Initialized Training...")
        self.training = True

        n_batches = 0
        total_loss = 0
        total_truth = 0

        try:
            while True:

                # Run the training graph nodes
                _, step_loss, step_summary, step_accuracy = sess.run([self.opt, self.loss_val, self.summary_op, self.accuracy])

                step += 1
                total_loss += step_loss
                n_batches += 1
                total_truth += step_accuracy

                writer.add_summary(step_summary, global_step=step)

                # Stepwise loss
                if ((step + 1) % self.skip_steps) == 0:
                    print("[LOSS - TRAIN (STEP)] at Step {0}: {1}".format(step, step_loss))
                    # Save learned weights
                    saver.save(sess, 'checkpoints/simpnet_train', step)

        except tf.errors.OutOfRangeError:
            pass

        # Overall loss
        print("[LOSS - TRAIN (EPOCH)] at Epoch {0}: {1}".format(epoch, total_loss/n_batches))

        # Epoch accuracy
        epoch_accuracy = (total_truth/(n_batches * self.batch_size)) * 100

        print("[ACCURACY - TRAIN] at epoch {0}: {1}".format(epoch, epoch_accuracy))
        print("[TIMING] Took {0} Seconds...".format(time.time() - start_time))

        return step


    def draw_class_maps(self):

        img = self.img
        logits = self.logits

        max_prop = logits[0]

        for i in logits.eval():
            candidate = logits[i]

            if candidate > max_prop:
                max_prop = candidate

        print("Maximum is: {0}".format(max_prop))

    def evaluate_network(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        print("[INFO]: Initialized Validation...")
        self.traininig = False

        total_truth = 0
        n_batches = 0

        try:
            while True:

                # Test the network
                batch_accuracy, step_summary = sess.run([self.accuracy, self.summary_op])
                total_truth += batch_accuracy
                n_batches += 1
                writer.add_summary(step_summary, global_step=step)

        except tf.errors.OutOfRangeError:
            pass

        validation_accuracy = (total_truth / (n_batches * self.batch_size)) * 100
        print("[ACCURACY - VALIDATION] at Epoch {0}: {1}".format(epoch, validation_accuracy))
        print("[TIMING] Took {0} Seconds...".format(time.time() - start_time))



    def train(self, n_epochs):

        safe_mkdir('checkpoints')
        safe_mkdir('checkpoints/simpnet_train')
        train_writer = tf.summary.FileWriter('./graphs/simpnet_train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./graphs/simpnet_test')

        with tf.Session() as sess:

            # Initialize the variables
            sess.run(tf.global_variables_initializer())

            # Check if there exists a training checkpoint
            saver = tf.train.Saver()

            # Restore the checkpoints (in case of any!)
            ckpt = tf.train.get_checkpoint_state('./checkpoints/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("[INFO]: Reloaded From Checkpoints...")
            else:
                print("[INFO]: No Checkpoint Found...")

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                # Train the model for one epoch
                step = self.train_network_one_epoch(
                   sess=sess,
                   init=self.train_init,
                   saver=saver,
                   writer=train_writer,
                   epoch=epoch,
                   step=step
                )

                # Evaluate the model after each epoch
                self.evaluate_network(
                    sess=sess,
                    init=self.test_init,
                    writer=test_writer,
                    epoch=epoch,
                    step=step
                )
        train_writer.close()
        test_writer.close()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    model = SimpNet()
    model.build_network_graph()
    model.train(n_epochs=50)
