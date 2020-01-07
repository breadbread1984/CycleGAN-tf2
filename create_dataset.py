#!/usr/bin/python3

from os import listdir, mkdir;
from os.path import join, exists;
import numpy as np;
import cv2;
import tensorflow as tf;

def parse_function_generator(img_shape):
  def parse_function(serialized_example):

    feature = tf.io.parse_single_example(
      serialized_example,
      features = {
        'data': tf.io.FixedLenFeature((), dtype = tf.string, default_value = '')
      }
    );
    data = tf.io.decode_raw(feature['data'], out_type = tf.uint8);
    data = tf.reshape(data, img_shape);
    data = tf.cast(data, dtype = tf.float32);
    return data, None;
  return parse_function;

def write_tfrecord(directory, filename):

  writer = tf.io.TFRecordWriter(filename);
  for filename in listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
      img = cv2.imread(join(directory, filename));
      if img is None: continue;
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img.tobytes()]))
        }
      ));
      writer.write(trainsample.SerializeToString());
    else: continue;
  writer.close();

if __name__ == "__main__":

  assert tf.executing_eagerly();
  if False == exists('dataset'): mkdir('dataset');
  write_tfrecord('A','dataset/A.tfrecord');
  write_tfrecord('B','dataset/B.tfrecord');
