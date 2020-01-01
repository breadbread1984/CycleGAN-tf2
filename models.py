#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_addons as tfa;

def Generator(input_filters, output_filters, inner_filters, blocks = 9):

  # input
  inputs = tf.keras.Input((None, None, input_filters));
  results = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT'))(inputs);
  results = tf.keras.layers.Conv2D(filters = inner_filters, kernel_size = (7,7), padding = 'valid')(results);
  results = tfa.layers.InstanceNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  # downsampling
  for i in range(2):
    m = 2**i;
    results = tf.keras.layers.Conv2D(filters = inner_filters * m * 2, kernel_size = (3,3), strides = (2,2), padding = 'same')(results);
    results = tfa.layers.InstanceNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
  # resnet blocks
  for i in range(blocks):
    short_circuit = results;
    results = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT'))(results);
    results = tf.keras.layers.Conv2D(filters = results.shape[-1], kernel_size = (3,3), padding = 'valid')(results);
    results = tfa.layers.InstanceNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Dropout(rate = 0.5)(results);
    results = tf.keras.layers.Add()([short_circuit, results]);
  # upsampling
  for i in range(2):
    m = 2**(2 - i);
    results = tf.keras.layers.Conv2DTranspose(filters = int(inner_filters * m / 2), kernel_size = (3,3), strides = (2,2), padding = 'same', output_padding = 1)(results);
    results = tfa.layers.InstanceNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
  # output
  results = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT'))(results);
  results = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = (7,7), padding = 'valid')(results);
  results = tf.keras.layers.Lambda(lambda x: tf.math.tanh(x))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Discriminator(input_filters, inner_filters, layers = 3):

  inputs = tf.keras.Input((None, None, input_filters));
  results = tf.keras.layers.Conv2D(filters = inner_filters, kernel_size = (4,4), strides = (2,2), padding = [(0,0),(1,1),(1,1),(0,0)])(inputs);
  results = tf.keras.layers.LeakyReLU(0.2)(results);
  for i in range(layers):
    m = min(2 ** i, 8);
    results = tf.keras.layers.Conv2D(filters = inner_filters * m, kernel_size = (4,4), strides = (2,2), padding = [(0,0),(1,1),(1,1),(0,0)])(results);
    results = tfa.layers.InstanceNormalization()(results);
    results = tf.keras.layers.LeakyReLU(0.2)(results);
  m = min(2 ** layers, 8);
  results = tf.keras.layers.Conv2D(filters = inner_filters * m, kernel_size = (4,4), strides = (1,1), padding = [(0,0),(1,1),(1,1),(0,0)])(results);
  results = tfa.layers.InstanceNormalization()(results);
  results = tf.keras.layers.LeakyReLU(0.2)(results);
  results = tf.keras.layers.Conv2D(filters = 1, kernel_size = (4,4), strides = (1,1), padding = [(0,0),(1,1),(1,1),(0,0)])(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

class CycleGAN(tf.keras.Model):

  def __init__(self, input_filters = 3, output_filters = 3, inner_filters = 64, blocks = 9, layers = 3, ** kwargs):

    super(CycleGAN, self).__init__(**kwargs);
    self.GA = Generator(input_filters = input_filters, output_filters = output_filters, inner_filters = inner_filters, blocks = blocks);
    self.GB = Generator(input_filters = output_filters, output_filters = input_filters, inner_filters = input_filters, blocks = blocks);
    self.DA = Discriminator(input_filters = output_filters, inner_filters = inner_filters, layers = layers);
    self.DB = Discriminator(input_filters = input_filters,  inner_filters = inner_filters, layers = layers);

  def call(self, inputs):

    real_A = inputs[0];
    real_B = inputs[1];
    fake_B = self.GA(real_A);
    rec_A = self.GB(fake_B);
    fake_A = self.GB(real_B);
    rec_B = self.GA(fake_A);
    return (fake_B, rec_A, fake_A, rec_B);

if __name__ == "__main__":
  assert True == tf.executing_eagerly();
  inputs = tf.keras.Input((480,640,3));
  generator = Generator(3, 512, 256);
  results = generator(inputs);
  generator.save('generator.h5');
  discriminator = Discriminator(512,256, 2);
  outputs = discriminator(results);
  discriminator.save('discriminator.h5');
  cyclegan = CycleGAN();
  cyclegan.save_weights('cyclegan.h5');
