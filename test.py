#!/usr/bin/python3

import cv2;
import tensorflow as tf;
import tensorflow_addons as tfa;
import tensorflow_datasets as tfds;
from download_dataset import parse_function_generator;

batch_size = 1;

def test():

  A = tfds.load(name = 'cycle_gan/horse2zebra', split = "testA", download = False).repeat(-1).map(parse_function_generator(isTrain = False)).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).__iter__();
  B = tfds.load(name = 'cycle_gan/horse2zebra', split = "testB", download = False).repeat(-1).map(parse_function_generator(isTrain = False)).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).__iter__();
  GA = tf.keras.models.load_model('models/GA.h5', compile = False, custom_objects = {'tf': tf});
  GB = tf.keras.models.load_model('models/GB.h5', compile = False, custom_objects = {'tf': tf});
  while True:
    imageA, _ = next(A);
    imageB, _ = next(B);
    fakeB = GA(imageA);
    fakeA = GB(imageB);
    cv2.imshow('real A', cv2.cvtColor(tf.clip_by_value((imageA[0] + 1.) * 127.5, clip_value_min = 0., clip_value_max = 255.).numpy().astype('uint8'), cv2.COLOR_RGB2BGR));
    cv2.imshow('fake B', cv2.cvtColor(tf.clip_by_value((fakeB[0] + 1.) * 127.5, clip_value_min = 0., clip_value_max = 255.).numpy().astype('uint8'), cv2.COLOR_RGB2BGR));
    cv2.imshow('real B', cv2.cvtColor(tf.clip_by_value((imageB[0] + 1.) * 127.5, clip_value_min = 0., clip_value_max = 255.).numpy().astype('uint8'), cv2.COLOR_RGB2BGR));
    cv2.imshow('fake A', cv2.cvtColor(tf.clip_by_value((fakeA[0] + 1.) * 127.5, clip_value_min = 0., clip_value_max = 255.).numpy().astype('uint8'), cv2.COLOR_RGB2BGR));
    cv2.waitKey();

if __name__ == "__main__":

  assert tf.executing_eagerly();
  test();
