#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;

def parse_function(feature):

  data = tf.cast(feature['image'], dtype = tf.float32) / 255.;
  return data, feature['label'];

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

