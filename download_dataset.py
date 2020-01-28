#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;

def parse_function_generator(isTrain = True):
  def parse_function(feature):

    if isTrain == True:
      # augmentation
      data = tf.image.resize(feature['image'], [tf.shape(feature['image'])[-3] + 30, tf.shape(feature['image'])[-2] + 30], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR);
      data = tf.image.random_crop(data, size = tf.shape(feature['image'])[-3:]);
      data = tf.image.random_flip_left_right(data);
    # normalize
    data = tf.cast(data, dtype = tf.float32) / 127.5 - 1.;
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

