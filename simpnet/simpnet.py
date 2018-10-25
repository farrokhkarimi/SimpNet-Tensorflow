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
from sklearn.metrics import confusion_matrix

class SimpNet(object):

    def __init__(self):

        # Dropout rate
        self.keep_prob = 1

        # Learning rate
        self.learning_rate = 0.001

        # Setup the data path (folder should contain train and test folders inside itself)
        self.data_path = '../images/'
        self.main_csv = '../Data_Entry_2017.csv'
        self.train_val_csv = shuffle_csv('../train_val_list.csv')
        self.test_csv = shuffle_csv('../test_list.csv')

        # Number of images in each batch
        self.batch_size = 12

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

        self.class_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

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
                
                # Normalize
                a = a / 255.0
                
                yield (np.array(a.flatten()), self.train_list.loc[self.train_list[self.train_list["Image Index"] == image].index.item(), self.all_labels].as_matrix())


    def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
        ''' 
        Parameters:
            correct_labels                  : These are your true classification categories.
            predict_labels                  : These are you predicted classification categories
            labels                          : This is a lit of labels which will be used to display the axix labels
            title='Confusion matrix'        : Title for your matrix
            tensor_name = 'MyFigure/image'  : Name for the output summay tensor

        Returns:
            summary: TensorFlow summary 

        Other itema to note:
            - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
            - Currently, some of the ticks dont line up due to rotations.
        '''
        cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
        if normalize:
            cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, copy=True)
            cm = cm.astype('int')

        np.set_printoptions(precision=2)
        ###fig, ax = matplotlib.figure.Figure()

        fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=4, va ='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
        fig.set_tight_layout(True)
        summary = tfplot.figure.to_summary(fig, tag=tensor_name)
        return summary


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
            stride=1,
            padding='SAME',
            scope_name='conv_2',
            keep_prob=self.keep_prob
        )

        conv3 = conv_bn_sc_relu(
            inputs=conv2,
            filters=CONV3_NUM_FILTERS,
            k_size=CONV3_FILTER_SIZE,
            stride=1,
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
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits)

            # Loss is mean of error on all dimensions
            self.loss_val = tf.reduce_mean(entropy, name='loss')


    def optimize(self):

        with tf.name_scope('optimizer'):
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate,
                name='SGD'
            ).minimize(self.loss_val, global_step=self.gstep)

    def summary(self):

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss_val)
            tf.summary.scalar('accuracy', self.step_accuracy)
            tf.summary.histogram('loss histogram', self.loss_val)
            self.summary_op = tf.summary.merge_all()

    def eval(self):

        with tf.name_scope('predict'):
            preds = tf.nn.sigmoid(self.logits)

            ground_truth  = tf.cast(tf.equal(self.label, tf.ones(shape=tf.shape(preds))), tf.float32)

            # Compare with 0.5 elementwise
            target = tf.fill(tf.shape(preds), 0.5, name='target')
            val0 = tf.fill(tf.shape(preds), 0.0, name='val0')
            val1 = tf.fill(tf.shape(preds), 1.0, name='val1')
            cond = tf.less(preds, target)
            preds = tf.where(cond, val0, val1)
            
            self.true_predicted_count = tf.reduce_sum(tf.cast(tf.equal(preds, ground_truth), tf.float32))
            self.total_count = tf.reduce_sum(ground_truth)
            
            self.step_accuracy = self.true_predicted_count

            # # Draw confusion matrix
            # checkpoint_dir = 'checkpoints/'
            # if(self.training == True):
            #     checkpoint_dir += 'simpnet_train'
            # else:
            #     checkpoint_dir += 'simpnet_test'
                    
            # img_d_summary_dir = os.path.join(checkpoint_dir, "summaries", "img")
            # img_d_summary_writer = tf.summary.FileWriter(img_d_summary_dir, sess.graph)
            # img_d_summary = self.plot_confusion_matrix(self.label, preds, self.class_list, tensor_name='dev/cm')
            # img_d_summary_writer.add_summary(img_d_summary, self.gstep)

    
    def train_network_one_epoch(self, sess, init, saver, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        print("[INFO]: Initialized Training...")
        self.training = True

        n_batches = 0
        total_loss = 0
        total_truth = 0
        total_total_count = 0

        try:
            while True:

                # Run the training graph nodes
                _, _, step_loss, step_summary, true_predicted_count, total_count = sess.run([self.step_accuracy, self.opt, self.loss_val, self.summary_op, self.true_predicted_count, self.total_count])

                step += 1
                total_loss += step_loss
                n_batches += 1
                total_truth += true_predicted_count
                total_total_count += total_count

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
        accuracy = (total_truth/total_total_count) * 100

        print("[ACCURACY - TRAIN] at epoch {0}: {1}".format(epoch, accuracy))
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
        total_total_count = 0
        n_batches = 0

        try:
            while True:

                # Test the network
                _, true_predicted_count, total_count, step_summary = sess.run([self.step_accuracy, self.true_predicted_count, self.total_count, self.summary_op])
                total_truth += true_predicted_count
                total_total_count += total_count
                n_batches += 1
                writer.add_summary(step_summary, global_step=step)

        except tf.errors.OutOfRangeError:
            pass

        accuracy = (total_truth / total_total_count) * 100
        print("[ACCURACY - VALIDATION] at Epoch {0}: {1}".format(epoch, accuracy))
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
