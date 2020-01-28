#!/usr/bin/python3

import os;
import tensorflow as tf;
from models import CycleGAN;

def save_model():

  cycleGAN = CycleGAN();
  optimizerGA = tf.keras.optimizers.Adam(
    tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries = [dataset_size * 100 + i * dataset_size * 100 / 4 for i in range(5)],
      values = list(reversed([i * 2e-4 / 5 for i in range(6)]))),
    beta_1 = 0.5);
  optimizerGB = tf.keras.optimizers.Adam(
    tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries = [dataset_size * 100 + i * dataset_size * 100 / 4 for i in range(5)],
      values = list(reversed([i * 2e-4 / 5 for i in range(6)]))),
    beta_1 = 0.5);
  optimizerDA = tf.keras.optimizers.Adam(
    tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries = [dataset_size * 100 + i * dataset_size * 100 / 4 for i in range(5)],
      values = list(reversed([i * 2e-4 / 5 for i in range(6)]))),
    beta_1 = 0.5);
  optimizerDB = tf.keras.optimizers.Adam(
    tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries = [dataset_size * 100 + i * dataset_size * 100 / 4 for i in range(5)],
      values = list(reversed([i * 2e-4 / 5 for i in range(6)]))),
    beta_1 = 0.5);
  checkpoint = tf.train.Checkpoint(GA = cycleGAN.GA, GB = cycleGAN.GB, DA = cycleGAN.DA, DB = cycleGAN.DB, 
                                   optimizerGA = optimizerGA, optimizerGB = optimizerGB, optimizerDA = optimizerDA, optimizerDB = optimizerDB);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  if False == os.path.exists('models'): os.mkdir('models');
  cycleGAN.GA.save(os.path.join('models', 'GA.h5'));
  cycleGAN.GB.save(os.path.join('models', 'GB.h5'));
  cycleGAN.DA.save(os.path.join('models', 'DA.h5'));
  cycleGAN.DB.save(os.path.join('models', 'DB.h5'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  save_model();
