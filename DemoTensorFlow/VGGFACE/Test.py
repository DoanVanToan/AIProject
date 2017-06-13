import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

filename = "faces94/test/anonym2/anonym2.1.jpg"

with tf.gfile.FastGFile(filename, 'rb') as f:
     image_data = f.read()

_decode_jpeg = tf.image.decode_jpeg(image_data, channels=3)

_decode_jpeg_resize = tf.image.resize_images(_decode_jpeg, [224, 224])

# cast = tf.cast(_decode_jpeg_resize, tf.uint8)
#
# _encode_jpeg_resize = tf.image.encode_jpeg(cast, format='rgb', quality=100)

sess = tf.Session()

image_data = sess.run(_decode_jpeg_resize)

image_data_array = []

image_data_array.append(image_data)


width = 224

height = 224

numClass = 3

image_input = tf.placeholder(tf.float32, shape=[None, height , width , 3], name='image_input')

label_input = tf.placeholder(tf.float32, shape=[None, numClass], name='label_input')

label_argmax = tf.argmax(label_input, dimension=1)

# is_training = False
#
conv1_1 = slim.conv2d(inputs=image_input, num_outputs=16, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv1_1')

conv1_2 = slim.conv2d(inputs=conv1_1, num_outputs=16, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv1_2')

pool1 = slim.max_pool2d(inputs=conv1_2, kernel_size=[2, 2], stride=2,scope='pool1')


conv2_1 = slim.conv2d(inputs=pool1, num_outputs=32, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv2_1')


conv2_2 = slim.conv2d(inputs=conv2_1, num_outputs=32, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv2_2')

pool2 = slim.max_pool2d(inputs=conv2_2, kernel_size=[2, 2], stride=2,scope='pool1')

conv3_1 = slim.conv2d(inputs=pool2, num_outputs=64, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv3_1')

conv3_2 = slim.conv2d(inputs=conv3_1, num_outputs=64, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv3_2')

conv3_3 = slim.conv2d(inputs=conv3_2, num_outputs=64, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv3_3')

pool3 = slim.max_pool2d(inputs=conv3_3, kernel_size=[2, 2], stride=2,scope='pool3')

conv4_1 = slim.conv2d(inputs=pool3, num_outputs=128, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv4_1')

conv4_2 = slim.conv2d(inputs=conv4_1, num_outputs=128, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv4_2')

conv4_3 = slim.conv2d(inputs=conv4_2, num_outputs=128, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv4_3')

pool4 = slim.max_pool2d(inputs=conv4_3, kernel_size=[2, 2], stride=2,scope='pool4')

conv5_1 = slim.conv2d(inputs=pool4, num_outputs=128, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv5_1')

conv5_2 = slim.conv2d(inputs=conv5_1, num_outputs=128, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv5_2')

conv5_3 = slim.conv2d(inputs=conv5_2, num_outputs=128, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv5_3')

pool5 = slim.max_pool2d(inputs=conv5_3, kernel_size=[2, 2], stride=2,scope='pool5')

flat = tf.reshape(pool5, [-1, 7 * 7 * 128])

fc6 = slim.fully_connected(inputs=flat, num_outputs=512, activation_fn=tf.nn.relu, scope='fc6')

dropout6 = slim.dropout(inputs=fc6, keep_prob=0.5, is_training=False, scope="dropout6")

fc7 = slim.fully_connected(inputs=dropout6, num_outputs=512, activation_fn=tf.nn.relu, scope='fc7')

dropout7 = slim.dropout(inputs=fc7, keep_prob=0.5, is_training=False, scope="dropout7")

logits = slim.fully_connected(inputs=dropout7, num_outputs=3, activation_fn=None, scope='logits')

predictions = tf.nn.softmax(logits=logits, name='predictions')

label_pred = tf.argmax(predictions, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label_input)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

correct_prediction = tf.equal(label_pred, label_argmax)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')





MODEL_DIR = 'savegraphface/'

file_name = "model.pb"

sess = tf.Session()

sess.run(tf.global_variables_initializer())

with gfile.FastGFile(MODEL_DIR + file_name, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

pred_1, predictions_1 = sess.run([label_pred, predictions],
                                     {image_input: image_data_array})

print("Pred : "+str(pred_1));