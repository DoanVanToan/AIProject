import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

filename = "./faces94/test/DataFace.tfrecords"


def oneHot(labels,numClass):

    oneHot = []
    for label in labels:
        vector = np.zeros(numClass,dtype=int);
        vector[label] = 1;
        oneHot.append(vector)
    return oneHot;

def shuffle(list_image, labels):
    arr = np.arange(length_img)

    np.random.shuffle(arr)

    Alist_image = []

    Alist_label = []

    for i in range(length_img):

        Alist_image.append(list_image[arr[i]])

        Alist_label.append(labels[arr[i]])

    return Alist_image,Alist_label

# Get label and bytes(image) in TFrecord
def read_and_decode_single_example(filename):

    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    imagesString = []
    labels = []
    i=0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        label = int(example.features.feature['label']
                     .int64_list
                     .value[0])

        img_string = (example.features.feature['encoded']
                      .bytes_list
                      .value[0])

        height = int(example.features.feature['height']
                    .int64_list
                    .value[0])
        width =  int(example.features.feature['width']
            .int64_list
            .value[0])

        imagesString.append(img_string)
        labels.append(label)
        #i+=1
        #if i == 100:
        #    break
        # break

    return  labels,imagesString

def showShape(target,tensor):
    # print(target)
    # print(tensor.get_shape())
    a = 1

labels , image_String = read_and_decode_single_example(filename)
# labels_test , image_String_test = read_and_decode_single_example(filename_test)
height = 39
width = 31

list_image = []

length_img = len(image_String)

sess = tf.Session()

for i in range(0,length_img):
    _decode_jpeg_data = tf.placeholder(dtype=tf.string)
    _decode_jpeg = tf.image.decode_jpeg(_decode_jpeg_data, channels=3)
    image_resize = sess.run(_decode_jpeg,feed_dict={_decode_jpeg_data : image_String[i]})
    list_image.append(image_resize)
    # list_image_test(image_resize)
    # plt.imshow(image_resize.astype(np.uint8), interpolation='nearest')
    # plt.show()

numClass = np.amax(labels) + 1

LableHot = oneHot(labels,numClass)

length_img = len(list_image)




image_input = tf.placeholder(tf.float32, shape=[None, height , width , 3], name='image_input')

label_input = tf.placeholder(tf.float32, shape=[None, numClass], name='label_input')

label_argmax = tf.argmax(label_input, dimension=1)

conv1 = slim.conv2d(inputs=image_input, num_outputs=20, kernel_size=[4, 4], stride=1,
                             activation_fn=tf.nn.relu, scope='conv1' , padding='VALID')
showShape("conv1 : ",conv1)
pool1 = slim.max_pool2d(inputs=conv1, kernel_size=[2, 2], stride=2,scope='pool1')
showShape("pool1 : ",pool1)
conv2 = slim.conv2d(inputs=pool1, num_outputs=40, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv2' , padding='VALID')
showShape("conv2 : ",conv2)
pool2 = slim.max_pool2d(inputs=conv2, kernel_size=[2, 2], stride=2,scope='pool2')
showShape("pool2 : ",pool2)
conv3 = slim.conv2d(inputs=pool2, num_outputs=60, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv3' , padding='VALID')
showShape("conv3 : ",conv3)
pool3 = slim.max_pool2d(inputs=conv3, kernel_size=[2, 2], stride=2,scope='pool3')
showShape("pool3 : ",pool3)
conv4 = slim.conv2d(inputs=pool3, num_outputs=60, kernel_size=[2, 2], stride=1,
                             activation_fn=tf.nn.relu, scope='conv4' , padding='VALID')
showShape("conv4 : ",conv4)
flat4 = tf.reshape(conv4, [-1, 2 * 1 * 60])
showShape("flat5 : ",flat4)
fc4 = slim.fully_connected(inputs=flat4, num_outputs=160, activation_fn=tf.nn.relu, scope='fc6')
showShape("fc4 : ",fc4)
logits = slim.fully_connected(inputs=fc4, num_outputs=int(numClass), activation_fn=None, scope='logits')
showShape("logits : ",logits)
predictions = tf.nn.softmax(logits=logits, name='predictions')
showShape("predictions : ",predictions)
label_pred = tf.argmax(predictions, dimension=1)
showShape("label_pred : ",label_pred)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label_input)
cost = tf.reduce_mean(cross_entropy,name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

correct_prediction = tf.equal(label_pred, label_argmax)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

saver = tf.train.Saver()

sess = tf.Session()

sess.run(tf.global_variables_initializer())

MODEL_DIR = 'savegraphface/'

file_name = "my_test_model.ckpt-100"

saver.restore(sess, MODEL_DIR+file_name)

accuracy_output_step, = sess.run([accuracy],
                                         feed_dict={image_input: list_image, label_input: LableHot})

print("accuracy : "+str(accuracy_output_step))