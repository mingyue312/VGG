"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import numpy as np
import glob

import vgg19_trainable as vgg19
import utils

IMG_FILE = '../train/*.jpg'
LABEL_FILE = '../train.csv'
NUM_EPOCH = 2
ITERATIONS_PER_EPOCH = 690   # Must have: 6900/ITERATION_PER_EPOCH = integer
BATCH_SIZE = 6900/ITERATIONS_PER_EPOCH


def parse_img(start, end):
    all_imgs = sorted(glob.glob(IMG_FILE))
    img_placeholder = np.empty([(end-start), 224, 224, 3])
    for i in range(start, end):
        img_placeholder[i-start,:,:,:] = utils.load_image(all_imgs[i])

    labels = np.genfromtxt(LABEL_FILE, delimiter=',')
    labels = labels[1:,1]
    labels = labels.astype(int)
    labels_one_hot = np.zeros([end-start, 8], dtype=np.int)
    for i in range(start, end):
        labels_one_hot[i-start, int(labels[i]-1)] = 1

    return img_placeholder, labels_one_hot


with tf.device('/cpu:0'):
    sess = tf.Session()

    val_img, val_label = parse_img(6900, 7000)

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, 8])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(vgg.fc8, true_out))
    prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(true_out, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    train_op = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss, var_list=[v for v in tf.all_variables() if v.name.startswith('fc8')])

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print vgg.get_var_count()

    sess.run(tf.initialize_all_variables())

    for k in range(0, NUM_EPOCH):
        for i in range(0, ITERATIONS_PER_EPOCH):
            # load batch images
            batch_img, batch_label = parse_img(i * BATCH_SIZE, (i+1) * BATCH_SIZE)

            train_acc = sess.run(accuracy, feed_dict={images:batch_img, true_out:batch_label, train_mode:False})
            valid_acc = sess.run(accuracy, feed_dict={images:val_img, true_out:val_label, train_mode:False})
            train_ce = sess.run(loss, feed_dict={images:batch_img, true_out:batch_label, train_mode:False})
            valid_ce = sess.run(loss, feed_dict={images:val_img, true_out:val_label, train_mode:False})
            print("Epoch: %d, Step: %d, train_acc: %g, val_acc: %g, train_ce: %g, val_ce: %g" % (k, i, train_acc, valid_acc, train_ce, valid_ce))

            # train:
            sess.run(train_op, feed_dict={images:batch_img, true_out:batch_label, train_mode:True})
        save_name = './save_epoch' + str(k) + '.npy'
        vgg.save_npy(sess, save_name)

