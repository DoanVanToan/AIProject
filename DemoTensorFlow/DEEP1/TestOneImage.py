import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

filename = "faces94/train/mytam/M16.jpg"

with tf.gfile.FastGFile(filename, 'rb') as f:
     image_data = f.read()

_decode_jpeg = tf.image.decode_jpeg(image_data, channels=3)

_decode_jpeg_resize = tf.image.resize_images(_decode_jpeg, [39, 31])

sess = tf.Session()

image_data = sess.run(_decode_jpeg_resize)

image_data_array = []

image_data_array.append(image_data)

height = 39

width = 31

numClass = 21

image_input = tf.placeholder(tf.float32, shape=[None, height , width , 3], name='image_input')

label_input = tf.placeholder(tf.float32, shape=[None, numClass], name='label_input')

label_argmax = tf.argmax(label_input, dimension=1)

conv1 = slim.conv2d(inputs=image_input, num_outputs=20, kernel_size=[4, 4], stride=1,
                             activation_fn=tf.nn.relu, scope='conv1' , padding='VALID')
pool1 = slim.max_pool2d(inputs=conv1, kernel_size=[2, 2], stride=2,scope='pool1')
conv2 = slim.conv2d(inputs=pool1, num_outputs=40, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv2' , padding='VALID')
pool2 = slim.max_pool2d(inputs=conv2, kernel_size=[2, 2], stride=2,scope='pool2')
conv3 = slim.conv2d(inputs=pool2, num_outputs=60, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv3' , padding='VALID')
pool3 = slim.max_pool2d(inputs=conv3, kernel_size=[2, 2], stride=2,scope='pool3')
conv4 = slim.conv2d(inputs=pool3, num_outputs=60, kernel_size=[2, 2], stride=1,
                             activation_fn=tf.nn.relu, scope='conv4' , padding='VALID')
flat4 = tf.reshape(conv4, [-1, 2 * 1 * 60])
fc4 = slim.fully_connected(inputs=flat4, num_outputs=160, activation_fn=tf.nn.relu, scope='fc6')
logits = slim.fully_connected(inputs=fc4, num_outputs=int(numClass), activation_fn=None, scope='logits')
predictions = tf.nn.softmax(logits=logits, name='predictions')
label_pred = tf.argmax(predictions, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label_input)
cost = tf.reduce_mean(cross_entropy,name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(label_pred, label_argmax)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

saver = tf.train.Saver()

sess = tf.Session()

sess.run(tf.global_variables_initializer())

MODEL_DIR = 'savegraphface/'

file_name = "my_test_model.ckpt-90"

saver.restore(sess, MODEL_DIR+file_name)

pred_1, predictions_1 = sess.run([label_pred, predictions],
                                     {image_input: image_data_array})

print("predictions : "+str(pred_1))
