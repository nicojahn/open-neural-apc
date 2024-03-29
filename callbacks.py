# -*- coding: utf-8 -*-
# Copyright (c) 2020-2021, Nico Jahn
# All rights reserved.
# pylint: disable=missing-module-docstring, too-few-public-methods
import tensorflow as tf


class StoppingAfterWarmup(tf.keras.callbacks.EarlyStopping):
    def __init__(
        self,
        monitor="accuracy",
        min_delta=0.005,
        patience=100,
        verbose=0,
        mode="max",
        baseline=None,
        restore_best_weights=False,
        baseline_accuracy=0.4,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
        )
        self.baseline_accuracy = baseline_accuracy
        self.active = False

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        if "accuracy" not in logs:
            return
        if logs["accuracy"] > self.baseline_accuracy or self.active:
            super().on_epoch_end(epoch, logs)
            self.active = True


class IncreaseEpochCustom(tf.keras.callbacks.Callback):
    def __init__(self, napc):
        self.napc = napc

    def on_epoch_end(self, epoch, logs=None):  # pylint: disable=unused-argument
        # Since Keras Progbar starts counting with 1, I have to add here 1
        self.napc.epoch = epoch + 1


# Tensorflow Keras ModelCheckpoint argument 'period' is deprecated
# Therefore, I'm doing it on my own
class SaveEveryNthEpochCustom(tf.keras.callbacks.Callback):
    def __init__(self, napc, save_steps):
        self.napc = napc
        self.save_steps = save_steps

    def on_epoch_end(self, epoch, logs=None):  # pylint: disable=unused-argument
        if self.napc.epoch % self.save_steps == 0:
            self.napc.save()
