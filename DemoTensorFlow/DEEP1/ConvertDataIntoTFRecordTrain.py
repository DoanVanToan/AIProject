from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import os
import random
import sys
import threading
import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

height = 39
width = 31

tf.app.flags.DEFINE_string('train_directory', './faces94/train',
                           'Training data directory')
tf.app.flags.DEFINE_string('output_directory', './faces94/train/',
                           'Output data directory')


tf.app.flags.DEFINE_integer('train_shards', 1,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 1,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 1,
                            'Number of threads to preprocess the images.')



# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('labels_file', './label.txt', 'Labels file')

numimage = 0

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'channels': _int64_feature(channels),
      'label': _int64_feature(label),
      'text': _bytes_feature(tf.compat.as_bytes(text)),
      'format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)

    self._jpeg_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)

    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
    self._decode_jpeg_resize = tf.image.resize_images(self._decode_jpeg, [height, width])
    self._cast = tf.cast(self._decode_jpeg_resize, tf.uint8)
    # self._grayscale = tf.image.rgb_to_grayscale(self._decode_jpeg_resize)
    self._encode_jpeg_resize = tf.image.encode_jpeg(self._cast, format='rgb', quality=100)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._encode_jpeg_resize,
                           feed_dict={self._decode_jpeg_data: image_data})
    # assert len(image.shape) == 3
    # assert image.shape[2] == 3
    return image


def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _process_image(filename, coder , numimage):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

  # print(type(image_data))
  # image_data = coder.resize_img(image_data)

  # Decode the RGB JPEG.
  image_data = coder.decode_jpeg(image_data)
  # print(image)

  # plt.imshow(image.astype(np.uint8), interpolation='nearest')
  # plt.show()
  # print(image)

  # Check that image converted to RGB
  # assert len(image.shape) == 3
  height = 50
  width = 50
  # print(str(height) + " / " + str(width))
  # assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]


  counter = 0


  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = 'DataFace' +".tfrecords"
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)

    numimage = 0

    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i] - 1
      text = texts[i]
      print(filename+" / "+str(text)+" / "+str(label))
      try:

        image_buffer, height, width = _process_image(filename, coder,numimage)

        # print(image_buffer)
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected eror while decoding %s.' % filename)
        continue

      example = _convert_to_example(filename, image_buffer, label,
                                    text, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):

  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # print(str(texts)+" / labels : "+str(labels))

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)

  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, labels_file):

  print('Determining list of input files and labels from %s.' % data_dir)

  # unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]

  unique_labels = []

  for k in tf.gfile.FastGFile(labels_file, 'r').readlines():
    if(k.strip() != ''):
      unique_labels.append(k.strip())


  labels = []
  filenames = []
  texts = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.

  print("unique : " + str(unique_labels))

  unique_labels_1 = []

  for text in unique_labels:

    text = data_dir +"/"+ text+"/"

    for root, dirs, files in os.walk(text, topdown=False):

      for path_image in files :

        path_image_1 = root + path_image

        matching_files = tf.gfile.Glob(path_image_1)

        labels.extend([label_index])

        texts.extend([text] * len(matching_files))

        filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file):
  """Process a complete data set and save it as a TFRecord.
  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """

  filenames, texts, labels = _find_image_files(directory, labels_file)
  _process_image_files(name, filenames, texts, labels, num_shards)


def main(unused_argv):

  # Run it!
  # _process_dataset('validation', FLAGS.validation_directory,
  #                  FLAGS.validation_shards, FLAGS.labels_file)
  _process_dataset('train', FLAGS.train_directory,
                   FLAGS.train_shards, FLAGS.labels_file)



if __name__ == '__main__':
  tf.app.run()