#!/usr/bin/python3

from os import listdir;
from os.path import join;
import numpy as np;
import cv2;
import tensorflow as tf;

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
  write_tfrecord('A','A.tfrecord');
  write_tfrecord('B','B.tfrecord');
