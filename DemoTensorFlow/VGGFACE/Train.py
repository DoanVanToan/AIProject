import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim


filename = "./faces94/train/DataFace.tfrecords"


# def trans_for_ohe(labels):
#     """Transform a flat list of labels to what one hot encoder needs."""
#     return np.array(labels).reshape(len(labels), -1)
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
    # print(tensor.shape)
    a = 1

labels , image_String = read_and_decode_single_example(filename)
# labels_test , image_String_test = read_and_decode_single_example(filename_test)

list_image = []
list_image_test = []

length_img = len(image_String)
# length_img_test = len(image_String_test)
# print(length_img)

minibatchsize = 32
height = 224
width = 224

sess = tf.Session()

for i in range(0,length_img):
    _decode_jpeg_data = tf.placeholder(dtype=tf.string)
    _decode_jpeg = tf.image.decode_jpeg(_decode_jpeg_data, channels=3)
    # _image_resize = tf.image.resize_images(_decode_jpeg, [height, width])
    image_resize = sess.run(_decode_jpeg,feed_dict={_decode_jpeg_data : image_String[i]})
    list_image.append(image_resize)
    # list_image_test(image_resize)
    # print("anh so :" + str(labels[i])+" / " +str(LableHot[i]))
    # plt.imshow(image_resize.astype(np.uint8), interpolation='nearest')
    # plt.show()


numClass = np.amax(labels)

LableHot = oneHot(labels,numClass)

length_img = len(list_image)

image_input = tf.placeholder(tf.float32, shape=[None, height , width , 3], name='image_input')

label_input = tf.placeholder(tf.float32, shape=[None, numClass], name='label_input')

label_argmax = tf.argmax(label_input, dimension=1)

# is_training = False
#
conv1_1 = slim.conv2d(inputs=image_input, num_outputs=2, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv1_1')
showShape("conv1_1 : ",conv1_1)

conv1_2 = slim.conv2d(inputs=conv1_1, num_outputs=2, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv1_2')
showShape("conv1_2 : ",conv1_2)
pool1 = slim.max_pool2d(inputs=conv1_2, kernel_size=[2, 2], stride=2,scope='pool1')
showShape("pool1 : ",pool1)

conv2_1 = slim.conv2d(inputs=pool1, num_outputs=4, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv2_1')
showShape("conv2_1 : ",conv2_1)

conv2_2 = slim.conv2d(inputs=conv2_1, num_outputs=4, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv2_2')
showShape("conv2_2 : ",conv2_2)
pool2 = slim.max_pool2d(inputs=conv2_2, kernel_size=[2, 2], stride=2,scope='pool1')
showShape("pool2 : ",pool2)
conv3_1 = slim.conv2d(inputs=pool2, num_outputs=8, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv3_1')
showShape("conv3_1 : ",conv3_1)
conv3_2 = slim.conv2d(inputs=conv3_1, num_outputs=8, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv3_2')
showShape("conv3_2 : ",conv3_2)
conv3_3 = slim.conv2d(inputs=conv3_2, num_outputs=8, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv3_3')
showShape("conv3_3 : ",conv3_3)
pool3 = slim.max_pool2d(inputs=conv3_3, kernel_size=[2, 2], stride=2,scope='pool3')
showShape("pool3 : ",pool3)
conv4_1 = slim.conv2d(inputs=pool3, num_outputs=16, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv4_1')
showShape("conv4_1 : ",conv4_1)
conv4_2 = slim.conv2d(inputs=conv4_1, num_outputs=16, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv4_2')
showShape("conv4_2 : ",conv4_2)
conv4_3 = slim.conv2d(inputs=conv4_2, num_outputs=16, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv4_3')
showShape("conv4_3 : ",conv4_3)
pool4 = slim.max_pool2d(inputs=conv4_3, kernel_size=[2, 2], stride=2,scope='pool4')
showShape("pool4 : ",pool4)
conv5_1 = slim.conv2d(inputs=pool4, num_outputs=16, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv5_1')
showShape("conv5_1 : ",conv5_1)
conv5_2 = slim.conv2d(inputs=conv5_1, num_outputs=16, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv5_2')
showShape("conv5_2 : ",conv5_2)
conv5_3 = slim.conv2d(inputs=conv5_2, num_outputs=16, kernel_size=[3, 3], stride=1,
                             activation_fn=tf.nn.relu, scope='conv5_3')
showShape("conv5_3 : ",conv5_3)
pool5 = slim.max_pool2d(inputs=conv5_3, kernel_size=[2, 2], stride=2,scope='pool5')
showShape("pool5 : ",pool5)
flat = tf.reshape(pool5, [-1, 7 * 7 * 16])
showShape("flat : ",flat)
fc6 = slim.fully_connected(inputs=flat, num_outputs=32, activation_fn=tf.nn.relu, scope='fc6')
showShape("fc6 : ",fc6)
dropout6 = slim.dropout(inputs=fc6, keep_prob=0.5, is_training=False, scope="dropout6")
showShape("dropout6 : ",dropout6)
fc7 = slim.fully_connected(inputs=dropout6, num_outputs=32, activation_fn=tf.nn.relu, scope='fc7')
showShape("fc7 : ",fc7)
dropout7 = slim.dropout(inputs=fc7, keep_prob=0.5, is_training=False, scope="dropout7")
showShape("dropout7 : ",dropout7)
logits = slim.fully_connected(inputs=dropout7, num_outputs=int(numClass), activation_fn=None, scope='logits')
showShape("logits : ",logits)
predictions = tf.nn.softmax(logits=logits, name='predictions')
showShape("predictions : ",predictions)
label_pred = tf.argmax(predictions, dimension=1)
showShape("label_pred : ",label_pred)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label_input)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

correct_prediction = tf.equal(label_pred, label_argmax)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

sess = tf.Session()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

save_dir = 'savegraphface/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(0,5):
    print(i)
    j = 0
    while(j * minibatchsize < length_img):
        print("........"+str(j))

        h = (((j+1)*minibatchsize)-1)

        u = length_img - h

        if(u < 0):
            h = length_img

        list_image_train = list_image[(j*minibatchsize) : h]

        LableHot = oneHot(labels,numClass)

        labels_image_train = LableHot[(j*minibatchsize) : h]

        image_output = sess.run(optimizer,feed_dict={image_input : list_image_train , label_input : labels_image_train })

        j += 1

    accuracy_output_step = sess.run(accuracy,
                                         feed_dict={image_input: list_image, label_input: LableHot})
    print("accuracy : "+str(accuracy_output_step))

    list_image, labels = shuffle(list_image, labels)

    tf.train.write_graph(sess.graph.as_graph_def(), save_dir, "model"+str(i)+".pb", as_text=False)


label_pred_output = sess.run(label_pred,feed_dict={image_input : list_image})

correct_prediction_output = sess.run(correct_prediction,feed_dict={image_input : list_image , label_input : LableHot })
