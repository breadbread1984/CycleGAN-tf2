#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from models import CycleGAN;
from create_dataset import parse_function_generator;
from download_dataset import parse_function;

batch_size = 4;
img_shape = (255,255,3);

def main():

  # models
  cycleGAN = CycleGAN();
  optimizer = tf.keras.optimizers.Adam(learning_rate = tf.keras.optimizers.schedulers.InverseTimeDecay(2e-4, 0.3, 1000 / batch_size));
  # load dataset
  '''
  A = tf.data.TFRecordDataset(os.path.join('dataset', 'A.tfrecord')).map(parse_function_generator(img_shape)).shuffle(batch_size).batch(batch_size).__iter__();
  B = tf.data.TFRecordDataset(os.path.join('dataset', 'B.tfrecord')).map(parse_function_generator(img_shape)).shuffle(batch_size).batch(batch_size).__iter__();
  '''
  A = tfds.load(name = 'cycle_gan/horse2zebra', split = "trainA", download = False).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).__iter__();
  B = tfds.load(name = 'cycle_gan/horse2zebra', split = "trainB", download = False).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).__iter__();
  # restore from existing checkpoint
  checkpoint = tf.train.Checkpoint(model = cycleGAN, optimizer = optimizer, optimizer_step = optimizer.iterations);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # create log
  log = tf.summary.create_file_writer('checkpoints');
  # train model
  g_loss = tf.keras.metrics.Mean(name = 'G loss', dtype = tf.float32);
  da_loss = tf.keras.metrics.Mean(name = 'DA loss', dtype = tf.float32);
  db_loss = tf.keras.metrics.Mean(name = 'DB loss', dtype = tf.float32);
  while True:
    imageA, _ = next(A);
    imageB, _ = next(B);
    with tf.GradientTape(persistent=True) as tape:
      outputs = cycleGAN((imageA, imageB));
      G_loss = cycleGAN.G_loss(outputs);    g_loss.update_state(G_loss);
      DA_loss = cycleGAN.DA_loss(outputs);  da_loss.update_state(DA_loss);
      DB_loss = cycleGAN.DB_loss(outputs);  db_loss.update_state(DB_loss);
    # calculate gradients
    ga_grads = tape.gradient(G_loss, cycleGAN.GA.trainable_variables);
    gb_grads = tape.gradient(G_loss, cycleGAN.GB.trainable_variables);
    da_grads = tape.gradient(DA_loss, cycleGAN.DA.trainable_variables);
    db_grads = tape.gradient(DB_loss, cycleGAN.DB.trainable_variables);
    # update weights
    optimizer.apply_gradients(zip(ga_grads, cycleGAN.GA.trainable_variables));
    optimizer.apply_gradients(zip(gb_grads, cycleGAN.GB.trainable_variables));
    optimizer.apply_gradients(zip(da_grads, cycleGAN.DA.trainable_variables));
    optimizer.apply_gradients(zip(db_grads, cycleGAN.DB.trainable_variables));
    if tf.equal(optimizer.iterations % 500, 0):
      with log.as_default():
        tf.summary.scalar('generator loss', g_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('discriminator A loss', da_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('discriminator B loss', db_loss.result(), step = optimizer.iterations);
      print('Step #%d G Loss: %.6f DA Loss: %.6f DB Loss: %.6f' % (optimizer.iterations, g_loss.result(), da_loss.result(), db_loss.result()));
      g_loss.reset_states();
      da_loss.reset_states();
      db_loss.reset_states();
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
