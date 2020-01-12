#!/usr/bin/python3

import os;
import tensorflow as tf;
from models import CycleGAN;

def save_model():

  cycleGAN = CycleGAN();
  optimizer = tf.keras.optimizers.Adam(2e-4);
  checkpoint = tf.train.Checkpoint(model = cycleGAN, optimizer = optimizer, optimizer_step = optimizer.iterations);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  cycleGAN.GA.save(os.path.join('models', 'GA.h5'));
  cycleGAN.GB.save(os.path.join('models', 'GB.h5'));
  cycleGAN.DA.save(os.path.join('models', 'DA.h5'));
  cycleGAN.DB.save(os.path.join('models', 'DB.h5'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  save_model();
