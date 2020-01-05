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

  IDENTITY_LOSS_WEIGHT = 0.5;
  CYCLE_LOSS_WEIGHT = 10.0;

  def __init__(self, input_filters = 3, output_filters = 3, inner_filters = 64, blocks = 9, layers = 3, ** kwargs):

    super(CycleGAN, self).__init__(**kwargs);
    self.GA = Generator(input_filters = input_filters, output_filters = output_filters, inner_filters = inner_filters, blocks = blocks);
    self.GB = Generator(input_filters = output_filters, output_filters = input_filters, inner_filters = input_filters, blocks = blocks);
    self.DA = Discriminator(input_filters = output_filters, inner_filters = inner_filters, layers = layers);
    self.DB = Discriminator(input_filters = input_filters,  inner_filters = inner_filters, layers = layers);
    self.bce = tf.keras.losses.BinaryCrossentropy();
    self.l1 = tf.keras.losses.MeanAbsoluteError();

  def call(self, inputs):

    real_A = inputs[0];
    real_B = inputs[1];
    # real_A => GA => fake_B   fake_B => DA => pred_fake_B
    #                          real_B => DA => pred_real_B
    # real_B => GB => fake_A   fake_A => DB => pred_fake_A
    #                          real_A => DB => pred_real_A
    fake_B = self.GA(real_A);
    pred_fake_B = self.DA(fake_B);
    pred_real_B = self.DA(real_B);
    rec_A = self.GB(fake_B);
    fake_A = self.GB(real_B);
    pred_fake_A = self.DB(fake_A);
    pred_real_A = self.DB(real_A);
    rec_B = self.GA(fake_A);
    
    return (real_A, real_B, fake_B, pred_fake_B, pred_real_B, rec_A, fake_A, pred_fake_A, pred_real_A, rec_B);

  def G_loss(self, inputs):
    
    (real_A, real_B, fake_B, pred_fake_B, pred_real_B, rec_A, fake_A, pred_fake_A, pred_real_A, rec_B) = inputs;
    # generated image should not deviate too much from origin image
    loss_idt_A = self.l1(fake_B, real_A);
    loss_idt_B = self.l1(fake_A, real_B);
    # distance from generated image to natural image
    loss_GA = self.bce(pred_fake_B, tf.ones_like(pred_B));
    loss_GB = self.bce(pred_fake_A, tf.ones_like(pred_A));
    # reconstruction loss
    loss_cycle_A = self.l1(rec_A, real_A);
    loss_cycle_B = self.l1(rec_B, real_B);
    
    return self.CYCLE_LOSS_WEIGHT * self.IDENTITY_LOSS_WEIGHT * (loss_idt_A + loss_idt_B) + \
           (loss_GA + loss_GB) + \
           self.CYCLE_LOSS_WEIGHT * (loss_cycle_A + loss_cycle_B);

  def DA_loss(self, inputs):

    (real_A, real_B, fake_B, pred_fake_B, pred_real_B, rec_A, fake_A, pred_fake_A, pred_real_A, rec_B) = inputs;
    real_loss = self.bce(pred_real_B, tf.ones_like(pred_real_B));
    fake_loss = self.bce(pred_fake_B, tf.zeros_like(pred_fake_B));
    return 0.5 * (real_loss + fake_loss);
    
  def DB_loss(self, inputs):

    (real_A, real_B, fake_B, pred_fake_B, pred_real_B, rec_A, fake_A, pred_fake_A, pred_real_A, rec_B) = inputs;
    real_loss = self.bce(pred_real_A, tf.ones_like(pred_real_A));
    fake_loss = self.bce(pred_fake_A, tf.zeros_like(pred_fake_A));
    return 0.5 * (real_loss + fake_loss);

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
