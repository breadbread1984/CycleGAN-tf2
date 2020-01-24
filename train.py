#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from models import CycleGAN;
from create_dataset import parse_function_generator;
from download_dataset import parse_function;

batch_size = 1;
img_shape = (255,255,3);

class LrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  BATCHES_PER_EPOCH = 1334 / batch_size;

  def __init__(self, initial_learning_rate, decay_from_epoch = 100, total_epoch = 200):

    self.initial_learning_rate = initial_learning_rate;
    self.decay_from_epoch = decay_from_epoch;
    self.total_epoch = total_epoch;

  def __call__(self, step):

    if step <= self.decay_from_epoch * self.BATCHES_PER_EPOCH:
      return self.initial_learning_rate;
    if self > self.total_epoch * self.BATCHES_PER_EPOCH:
      return 0.;
    return (self.total_epoch * self.BATCHES_PER_EPOCH - step) / ((self.total_epoch - self.decay_from_epoch) * self.BATCHES_PER_EPOCH) * self.initial_learning_rate;

def main():

  # models
  cycleGAN = CycleGAN();
  optimizer = tf.keras.optimizers.Adam(LrSchedule(initial_learning_rate = 2e-4), beta_1 = 0.5);
  # load dataset
  '''
  A = tf.data.TFRecordDataset(os.path.join('dataset', 'A.tfrecord')).map(parse_function_generator(img_shape)).shuffle(batch_size).batch(batch_size).__iter__();
  B = tf.data.TFRecordDataset(os.path.join('dataset', 'B.tfrecord')).map(parse_function_generator(img_shape)).shuffle(batch_size).batch(batch_size).__iter__();
  '''
  A = tfds.load(name = 'cycle_gan/horse2zebra', split = "trainA", download = False).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).__iter__();
  B = tfds.load(name = 'cycle_gan/horse2zebra', split = "trainB", download = False).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).__iter__();
  testA = tfds.load(name = 'cycle_gan/horse2zebra', split = 'testA', download = False).repeat(-1).map(parse_function).batch(1).__iter__();
  testB = tfds.load(name = 'cycle_gan/horse2zebra', split = 'testB', download = False).repeat(-1).map(parse_function).batch(1).__iter__();
  # restore from existing checkpoint
  checkpoint = tf.train.Checkpoint(model = cycleGAN, optimizer = optimizer, optimizer_step = optimizer.iterations);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # create log
  log = tf.summary.create_file_writer('checkpoints');
  # train model
  avg_g_loss = tf.keras.metrics.Mean(name = 'G loss', dtype = tf.float32);
  avg_da_loss = tf.keras.metrics.Mean(name = 'DA loss', dtype = tf.float32);
  avg_db_loss = tf.keras.metrics.Mean(name = 'DB loss', dtype = tf.float32);
  while True:
    imageA, _ = next(A);
    imageB, _ = next(B);
    with tf.GradientTape(persistent = True) as tape:
      outputs = cycleGAN((imageA, imageB));
      G_loss = cycleGAN.G_loss(outputs);
      DA_loss = cycleGAN.DA_loss(outputs);
      DB_loss = cycleGAN.DB_loss(outputs);
    # calculate discriminator gradients
    da_grads = tape.gradient(DA_loss, cycleGAN.DA.trainable_variables);
    db_grads = tape.gradient(DB_loss, cycleGAN.DB.trainable_variables);
    avg_da_loss.update_state(DA_loss);
    avg_db_loss.update_state(DB_loss);
    # update discriminator weights
    optimizer.apply_gradients(zip(da_grads, cycleGAN.DA.trainable_variables));
    optimizer.apply_gradients(zip(db_grads, cycleGAN.DB.trainable_variables));
    # calculate generator gradients
    ga_grads = tape.gradient(G_loss, cycleGAN.GA.trainable_variables);
    gb_grads = tape.gradient(G_loss, cycleGAN.GB.trainable_variables);
    avg_g_loss.update_state(G_loss);
    # update generator weights
    optimizer.apply_gradients(zip(ga_grads, cycleGAN.GA.trainable_variables));
    optimizer.apply_gradients(zip(gb_grads, cycleGAN.GB.trainable_variables));
    if tf.equal(optimizer.iterations % 500, 0):
      imageA, _ = next(testA);
      imageB, _ = next(testB);
      outputs = cycleGAN((imageA, imageB));
      real_A = tf.cast(tf.clip_by_value(imageA * 255., clip_value_min = 0., clip_value_max = 255.), dtype = tf.uint8);
      real_B = tf.cast(tf.clip_by_value(imageB * 255., clip_value_min = 0., clip_value_max = 255.), dtype = tf.uint8);
      fake_B = tf.cast(tf.clip_by_value(outputs[1] * 255., clip_value_min = 0., clip_value_max = 255.), dtype = tf.uint8);
      fake_A = tf.cast(tf.clip_by_value(outputs[7] * 255., clip_value_min = 0., clip_value_max = 255.), dtype = tf.uint8);
      with log.as_default():
        tf.summary.scalar('generator loss', avg_g_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('discriminator A loss', avg_da_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('discriminator B loss', avg_db_loss.result(), step = optimizer.iterations);
        tf.summary.image('real A', real_A, step = optimizer.iterations);
        tf.summary.image('fake B', fake_B, step = optimizer.iterations);
        tf.summary.image('real B', real_B, step = optimizer.iterations);
        tf.summary.image('fake A', fake_A, step = optimizer.iterations);
      print('Step #%d G Loss: %.6f DA Loss: %.6f DB Loss: %.6f lr: %.6f' % \
            (optimizer.iterations, avg_g_loss.result(), avg_da_loss.result(), avg_db_loss.result(), optimizer._hyper['learning_rate']));
      avg_g_loss.reset_states();
      avg_da_loss.reset_states();
      avg_db_loss.reset_states();
    if tf.equal(optimizer.iterations % 10000, 0):
      # save model
      checkpoint.save(os.path.join('checkpoints', 'ckpt'));
    if G_loss < 0.01 and DA_loss < 0.01 and DB_loss < 0.01: break;
  # save the network structure with weights
  if False == os.path.exists('models'): os.mkdir('models');
  cycleGAN.GA.save(os.path.join('models', 'GA.h5'));
  cycleGAN.GB.save(os.path.join('models', 'GB.h5'));
  cycleGAN.DA.save(os.path.join('models', 'DA.h5'));
  cycleGAN.DB.save(os.path.join('models', 'DB.h5'));

if __name__ == "__main__":
    
  assert True == tf.executing_eagerly();
  main();
