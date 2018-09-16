# ========================================
# [] File Name : simpnet.py
#
# [] Creation Date : Aug 2018
#
# [] Author : Ali Gholami
#
# ========================================

import tensorflow as tf
from cnn_util import conv_bn_sc_relu, saf_pool
from cnn_config import *
# from data_util import *
import time
import utils as utils
import numpy as np
import os

class SimpNet(object):

    def __init__(self):

        # Dropout rate
        self.keep_prob = 0.7

        # Learning rate
        self.learning_rate = 0.001

        # Setup the data path (folder should contain train and test folders inside itself)
        self.data_path = './data'

        # Number of images in each batch
        self.batch_size = 128

        # Number of classes
        self.n_classes = 10

        # Global step (times the graph seen the data)
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Shows the overall graph status (Trainig vs Testing)
        self.training = True
        
        # Which steps show the loss in each epoch
        self.skip_steps = 10

        self.n_test = 10000


    def get_data(self):
        
        with tf.name_scope('data'):
            
            train_data, test_data =  utils.get_mnist_dataset(self.batch_size)
            iterator = tf.data.Iterator.from_structure(output_types=train_data.output_types, output_shapes=train_data.output_shapes)

            print('HELOOO')
            img, self.label = iterator.get_next()
            print("shape before: ", img.shape)
            self.img = tf.reshape(img, [-1, CNN_INPUT_HEIGHT, CNN_INPUT_WIDTH, CNN_INPUT_CHANNELS])
            print("shape after: ", img.shape)

            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)

    def build_network_graph(self):

        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()
    
    def inference(self):
        
        print("INPUT SHAPE: ", self.img.shape)

        conv1 = conv_bn_sc_relu(
            inputs=self.img,
            filters=CONV1_NUM_FILTERS,
            k_size=CONV1_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_1'
        )

        conv2 = conv_bn_sc_relu(
            inputs=conv1,
            filters=CONV2_NUM_FILTERS,
            k_size=CONV2_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_2'
        )

        conv3 = conv_bn_sc_relu(
            inputs=conv2,
            filters=CONV3_NUM_FILTERS,
            k_size=CONV3_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_3'
        )

        conv4 = conv_bn_sc_relu(
            inputs=conv3, 
            filters=CONV4_NUM_FILTERS,
            k_size=CONV4_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_4'
        )

        conv5 = conv_bn_sc_relu(
            inputs=conv4,
            filters=CONV5_NUM_FILTERS,
            k_size=CONV5_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_5'
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
            scope_name='conv_6'
        )

        conv7 = conv_bn_sc_relu(
            inputs=conv6,
            filters=CONV7_NUM_FILTERS,
            k_size=CONV7_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_7'
        )

        conv8 = conv_bn_sc_relu(
            inputs=conv7,
            filters=CONV8_NUM_FILTERS,
            k_size=CONV8_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_8'
        )

        conv9 = conv_bn_sc_relu(
            inputs=conv8,
            filters=CONV9_NUM_FILTERS,
            k_size=CONV9_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_9'
        )

        conv10 = conv_bn_sc_relu(
            inputs=conv9,
            filters=CONV10_NUM_FILTERS,
            k_size=CONV10_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_10'
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
            scope_name='conv_11'
        )

        conv12 = conv_bn_sc_relu(
            inputs=conv11,
            filters=CONV11_NUM_FILTERS,
            k_size=CONV11_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_12',
        )

        conv13 = conv_bn_sc_relu(
            inputs=conv12, 
            filters=CONV13_NUM_FILTERS,
            k_size=CONV13_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_13'
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
            
            print('predictions shape {0}'.format(preds.shape))
            print('labels shape {0}'.format(self.label.shape))

            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))

            # Summation of all probabilities of all correct predictions
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    
    def train_network_one_epoch(self, sess, init, saver, writer, epoch, step):
        start_time = time.time()
        
        # Initialize training (ready data)
        sess.run(init)
        self.training = True

        n_batches = 0
        total_loss = 0

        try:
            while True:

                # Run the training graph nodes
                _, step_loss, step_summary = sess.run([self.opt, self.loss_val, self.summary_op])

                step += 1
                total_loss += step_loss
                n_batches += 1
                writer.add_summary(step_summary, global_step=step)

                # if step + 1 % self.skip_steps == 0:
                print("loss at step {0}: {1}".format(step, step_loss))

        except tf.errors.OutOfRangeError:
            pass

        # Save learned weights
        saver.save(sess, 'checkpoints/simpnet_train', step)

        print("Average loss at epoch {0}: {1}".format(epoch, total_loss/n_batches))
        print("Took {0} seconds...".format(time.time() - start_time))
    
        return step

    def evaluate_network(self, sess, init, writer, epoch, step):

        start_time = time.time()

        # Initialize the testing (ready test data)
        sess.run(init)
        self.traininig = False

        total_truth = 0 

        try:
            while True:

                # Test the network 
                batch_accuracy, step_summary = sess.run([self.accuracy, self.summary_op])

                total_truth += batch_accuracy
                writer.add_summary(step_summary, global_step=step)

        except tf.errors.OutOfRangeError:
            pass


        print("Accuracy at step {0}: {1}".format(epoch, total_truth/self.n_test))


    def train(self, n_epochs):
    
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/simpnet_train')
        writer = tf.summary.FileWriter('./graphs/simpnet', graph=tf.get_default_graph())

        with tf.Session() as sess:
            
            # Initialize the variables
            sess.run(tf.global_variables_initializer())

            # Check if there exists a training checkpoint
            saver = tf.train.Saver()
            
            # Restore the checkpoints (in case of any!)
            # saver.restore(sess, os.path.dirname('checkpoints/simpnet_train/checkpoint'))
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                # Train the model for one epoch
                step = self.train_network_one_epoch(
                    sess=sess,
                    init=self.train_init,
                    saver=saver,
                    writer=writer,
                    epoch=epoch,
                    step=step
                )
        
                # Evaluate the model after each epoch
                self.evaluate_network(
                    sess=sess, 
                    init=self.test_init,
                    writer=writer,
                    epoch=epoch,
                    step=step
                )
        
        writer.close()

if __name__ == '__main__':
    model = SimpNet()
    model.build_network_graph()
    model.train(n_epochs=20)
