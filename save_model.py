#!/usr/bin/python3

import os;
import tensorflow as tf;
from models import CycleGAN;

def save_model():

  cycleGAN = CycleGAN();
  optimizerGA = tf.keras.optimizers.Adam(2e-4);
  optimizerGB = tf.keras.optimizers.Adam(2e-4);
  optimizerDA = tf.keras.optimizers.Adam(2e-4);
  optimizerDB = tf.keras.optimizers.Adam(2e-4);
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
