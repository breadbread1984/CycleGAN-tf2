#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;

def parse_function_generator(image_shape = (256,256,3), isTrain = True)
  def parse_function(feature):

    if isTraint == True:
      # augmentation
      data = tf.image.resize(data, [image_shape[0] + 30,image_shape[1] + 30], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR);
      data = tf.image.random_crop(data, size = image_shape);
      data = tf.image.random_flip_left_right(data);
    # normalize
    data = tf.cast(image, tf.float32) / 127.5 - 1;
    return data, feature['label'];
  return parse_function;

def download():

  # load dataset
  dataset_builder = tfds.builder('cycle_gan/horse2zebra');
  dataset_builder.download_and_prepare();
  # try to load the dataset once
  train_a = tfds.load(name = 'cycle_gan/horse2zebra', split = "trainA", download = False);
  train_b = tfds.load(name = 'cycle_gan/horse2zebra', split = "trainB", download = False);
  test_a = tfds.load(name = 'cycle_gan/horse2zebra', split = "testA", download = False);
  test_b = tfds.load(name = 'cycle_gan/horse2zebra', split = "testB", download = False);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  download();

