#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_dataset as tfds;

def main():

  # load dataset
  dataset_builder = tfds.builder('horse2zebra');
  dataset_builder.download_and_prepare();
  # try to load the dataset once
  train_a = tfds.load(name = 'horse2zebra', split = tfds.Split.TRAINA, download = False);
  train_b = tfds.load(name = 'horse2zebra', split = tfds.Split.TRAINB, download = False);
  test_a = tfds.load(name = 'horse2zebra', split = tfds.Split.TESTA, download = False);
  test_b = tfds.load(name = 'horse2zebra', split = tfds.Split.TESTB, download = False);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();

