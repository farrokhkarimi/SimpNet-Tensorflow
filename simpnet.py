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
from data_util import *
import time

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
        self.n_classes = 15

        # Global step (times the graph seen the data)
        self.gstep = 0

        # Shows the overall graph status (Trainig vs Testing)
        self.training = True
        
        # Which steps show the loss in each epoch
        self.skip_steps = 10

        self.n_test = 1000

    def get_data(self):
        
        with tf.name_scope('data'):
            
            train_data, test_data = load_image_data(dir_path=self.data_path, batch_size=self.batch_size)
            iterator = tf.data.Iterator.from_structure(output_types=train_data.output_types, output_shapes=train_data.output_shapes)

            img, self.label = iterator.get_next()

            self.img = tf.reshape(img, [-1, CNN_INPUT_HEIGHT, CNN_INPUT_WIDTH, CNN_INPUT_CHANNELS])

            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)

    
    def inference(self):

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
            name='fully_connected'
        )
    

    def loss(self):

        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
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
            tf.summary.scalar(name='loss', tensor=self.loss_val)
            tf.summary.scalar(name='accuracy', tensor=self.accuracy)
            tf.summary.histogram(name='loss histogram', tensor=self.loss_val)
            self.summary_op = tf.summary.merge_all()
            
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

                total_loss += step_loss
                n_batches += 1
                writer.add_summary(step_summary, global_step=self.gstep)

                if step + 1 % self.skip_steps == 0:
                    print("loss at step {0}: {1}".format(step, step_loss))

        except tf.errors.OutOfRangeError as err:
            pass

        print("Average loss at epoch {0}: {1}".format(epoch, total_loss/n_batches))
        print("Took {0} seconds...".format(time.time() - start_time))
    
        return step

    def evaluate_network(self, sess, init, saver, writer, epoch, step):

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
                writer.add_summary(step_summary, global_step=self.gstep)

        except tf.errors.OutOfRangeError as err:
            pass

        print("Accuracy at step {0}: {1}".format(epoch, total_truth/self.n_test))


    def train(self, epochs):
    
        # Load the model

        with tf.Session() as sess:

            
            step = self.gstep.eval()

            for epoch in range(epochs):
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

            

