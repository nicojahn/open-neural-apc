# Copyright (c) 2020, Nico Jahn
# All rights reserved.

import tensorflow as tf

class StoppingAfterWarmup(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='accuracy', min_delta=0.005, patience=100, verbose=0, mode='max',\
                 baseline=None, restore_best_weights=False, baseline_accuracy = 0.4):
        super(StoppingAfterWarmup, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline, restore_best_weights=restore_best_weights)
        self.baseline_accuracy = baseline_accuracy
        self.active = False

    def on_epoch_end(self, epoch, logs={}):
        if not "accuracy" in logs: return
        if logs["accuracy"] > self.baseline_accuracy or self.active:
            super().on_epoch_end(epoch, logs)
            self.active = True

class IncreaseEpochCustom(tf.keras.callbacks.Callback):
    def __init__(self,napc):
        self.napc = napc
    def on_epoch_end(self, epoch, logs=None):
        # Since Keras Progbar starts counting with 1, I have to add here 1 
        self.napc.epoch = epoch+1

# Tensorflow Keras ModelCheckpoint argument 'period' is deprecated
# Therefore, I'm doing it on my own
class SaveEveryNthEpochCustom(tf.keras.callbacks.Callback):
    def __init__(self,napc,save_steps):
        self.napc = napc
        self.save_steps = save_steps
    def on_epoch_end(self, epoch, logs=None):
        if self.napc.epoch%self.save_steps == 0:
            self.napc.save()