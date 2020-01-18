#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_addons as tfa;

def Generator(input_filters = 3, output_filters = 3, inner_filters = 64, blocks = 9):

  # input
  inputs = tf.keras.Input((None, None, input_filters));
  results = tf.keras.layers.Conv2D(filters = inner_filters, kernel_size = (7,7), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(inputs);
  results = tfa.layers.InstanceNormalization(axis = -1)(results);
  results = tf.keras.layers.ReLU()(results);
  # downsampling
  # 128-256
  for i in range(2):
    m = 2**(i + 1);
    results = tf.keras.layers.Conv2D(filters = inner_filters * m, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
    results = tfa.layers.InstanceNormalization(axis = -1)(results);
    results = tf.keras.layers.ReLU()(results);
  # resnet blocks
  # 256
  for i in range(blocks):
    short_circuit = results;
    results = tf.keras.layers.Conv2D(filters = inner_filters * 4, kernel_size = (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
    results = tfa.layers.InstanceNormalization(axis = -1)(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = inner_filters * 4, kernel_size = (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
    results = tfa.layers.InstanceNormalization(axis = -1)(results);
    results = tf.keras.layers.Concatenate()([short_circuit, results]);
  # upsampling
  # 128-64
  for i in range(2):
    m = 2**(1 - i);
    results = tf.keras.layers.Conv2DTranspose(filters = inner_filters * m, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
    results = tfa.layers.InstanceNormalization(axis = -1)(results);
    results = tf.keras.layers.ReLU()(results);
  # output
  results = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = (7,7), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
  results = tfa.layers.InstanceNormalization(axis = -1)(results);
  results = tf.keras.layers.Lambda(lambda x: tf.math.tanh(x))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Discriminator(input_filters, inner_filters, layers = 3):

  inputs = tf.keras.Input((None, None, input_filters));
  results = tf.keras.layers.Conv2D(filters = inner_filters, kernel_size = (4,4), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(inputs);
  results = tf.keras.layers.LeakyReLU(0.2)(results);
  # 128-256-512
  for i in range(layers):
    m = min(2 ** (i + 1), 8);
    results = tf.keras.layers.Conv2D(filters = inner_filters * m, kernel_size = (4,4), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
    results = tfa.layers.InstanceNormalization(axis = -1)(results);
    results = tf.keras.layers.LeakyReLU(0.2)(results);
  m = min(2 ** layers, 8); # 512
  results = tf.keras.layers.Conv2D(filters = inner_filters * m, kernel_size = (4,4), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
  results = tfa.layers.InstanceNormalization(axis = -1)(results);
  results = tf.keras.layers.LeakyReLU(0.2)(results);
  results = tf.keras.layers.Conv2D(filters = 1, kernel_size = (4,4), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

class CycleGAN(tf.keras.Model):

  def __init__(self, input_filters = 3, output_filters = 3, inner_filters = 64, blocks = 9, layers = 3, ** kwargs):

    super(CycleGAN, self).__init__(**kwargs);
    self.GA = Generator(input_filters = input_filters, output_filters = output_filters, inner_filters = inner_filters, blocks = blocks);
    self.GB = Generator(input_filters = output_filters, output_filters = input_filters, inner_filters = input_filters, blocks = blocks);
    self.DA = Discriminator(input_filters = output_filters, inner_filters = inner_filters, layers = layers);
    self.DB = Discriminator(input_filters = input_filters,  inner_filters = inner_filters, layers = layers);
    self.mse = tf.keras.losses.MeanSquaredError();
    self.l1 = tf.keras.losses.MeanAbsoluteError();

  def call(self, inputs):

    real_A = inputs[0];
    real_B = inputs[1];
    # real_A => GA => fake_B   fake_B => DA => pred_fake_B
    #                          real_B => DA => pred_real_B
    # real_B => GB => fake_A   fake_A => DB => pred_fake_A
    #                          real_A => DB => pred_real_A
    fake_B = self.GA(real_A);
    idt_B = self.GA(real_B);
    pred_fake_B = self.DA(fake_B);
    pred_real_B = self.DA(real_B);
    rec_A = self.GB(fake_B);
    fake_A = self.GB(real_B);
    idt_A = self.GB(real_A);
    pred_fake_A = self.DB(fake_A);
    pred_real_A = self.DB(real_A);
    rec_B = self.GA(fake_A);
    
    return (real_A, fake_B, idt_B, pred_fake_B, pred_real_B, rec_A, real_B, fake_A, idt_A, pred_fake_A, pred_real_A, rec_B);

  def G_loss(self, inputs):
    
    (real_A, fake_B, idt_B, pred_fake_B, pred_real_B, rec_A, real_B, fake_A, idt_A, pred_fake_A, pred_real_A, rec_B) = inputs;
    # generated image should not deviate too much from origin image
    loss_idt_A = self.l1(real_A, idt_A);
    loss_idt_B = self.l1(real_B, idt_B);
    # distance from generated image to natural image
    loss_GA = self.mse(tf.ones_like(pred_fake_B), pred_fake_B);
    loss_GB = self.mse(tf.ones_like(pred_fake_A), pred_fake_A);
    # reconstruction loss
    loss_cycle_A = self.l1(real_A, rec_A);
    loss_cycle_B = self.l1(real_B, rec_B);
    
    return 5 * (loss_idt_A + loss_idt_B) + (loss_GA + loss_GB) + 10 * (loss_cycle_A + loss_cycle_B);

  def DA_loss(self, inputs):

    (real_A, fake_B, idt_B, pred_fake_B, pred_real_B, rec_A, real_B, fake_A, idt_A, pred_fake_A, pred_real_A, rec_B) = inputs;
    real_loss = self.mse(tf.ones_like(pred_real_B), pred_real_B);
    fake_loss = self.mse(tf.zeros_like(pred_fake_B), pred_fake_B);
    return real_loss + fake_loss;

  def DB_loss(self, inputs):

    (real_A, fake_B, idt_B, pred_fake_B, pred_real_B, rec_A, real_B, fake_A, idt_A, pred_fake_A, pred_real_A, rec_B) = inputs;
    real_loss = self.mse(tf.ones_like(pred_real_A), pred_real_A);
    fake_loss = self.mse(tf.zeros_like(pred_fake_A), pred_fake_A);
    return real_loss + fake_loss;

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
