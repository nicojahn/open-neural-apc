# -*- coding: utf-8 -*-
# Copyright (c) 2020-2021, Nico Jahn
# All rights reserved.
# pylint: disable=no-name-in-module, line-too-long
"""Neural Network class with architecture and loss functions.

The NeuralAPC class is aimed for the creation and restoration of the Neural Network. The main interfaces are the constructor and the compile(), save() and load_model() methods. The class is structured in 3 parts (top to bottom): The main interfaces, utility functions, the loss/metric functions.

  Typical usage example:

  napc = NeuralAPC(model_parameter, training_parameter)
  napc.compile()
  napc.save()

  or

  napc = NeuralAPC(model_parameter, training_parameter)
  napc.load_model(10000, "models/")
  napc.compile()
"""
# pylint: enable=line-too-long
import datetime
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.compat.v1.keras.layers import CuDNNLSTM  # pylint: disable=import-error
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import model_from_json

from callbacks import IncreaseEpochCustom
from callbacks import SaveEveryNthEpochCustom


class NeuralAPC:
    def __init__(self, *args, verbose=0):
        (self._model_parameter, self._training_parameter) = args
        self._model_path = "./models/%s/" % str(datetime.datetime.now()).replace(
            " ", "_"
        )
        self._verbose = verbose

        # setting the precision to mixed if float16 is desired in training_parameter
        self._set_precision(
            self._training_parameter["calculation_dtype"],
            self._training_parameter["calculation_epsilon"],
        )

        # assemble network
        self._model = None
        self._create_model()
        self._add_input()
        self._add_core()
        self._add_output()

        if self._verbose:
            # print the model properties
            self._model.summary()

        # set an epoch
        self._epoch = 0

        clip_gradient = "optimizer_clip_parameter" in self._training_parameter
        is_half_precision = "float16" in self._training_parameter["calculation_dtype"]

        optimizer_clip_parameter = {}
        if (
            clip_gradient
            and self._training_parameter["optimizer_clip_parameter"] is not None
            and not is_half_precision
        ):
            optimizer_clip_parameter = self._training_parameter[
                "optimizer_clip_parameter"
            ]

        self._optimizer = keras.optimizers.Adam(
            self._training_parameter["learning_rate"],
            **self._training_parameter["optimizer_parameter"],
            **optimizer_clip_parameter
        )

        # helper for the loss
        self._tf_zero = K.cast(0.0, dtype=K.floatx())
        self._tf_one = K.cast(1.0, dtype=K.floatx())
        self._aux_scale = K.cast(
            self._training_parameter["aux_scale"], dtype=K.floatx()
        )
        self._slack = K.cast(
            self._training_parameter["accuracy_error_niveau"], dtype=K.floatx()
        )

        # creating list with some important callbacks
        self._callbacks = []
        self._add_callbacks()

    def compile(self):
        self._model.compile(
            loss=self.loss, optimizer=self._optimizer, metrics=[self.accuracy]
        )

    def save(self):
        # create model directory first
        os.makedirs(self._model_path, exist_ok=True)
        # serialize model to JSON
        model_json = self._model.to_json()
        with open("%smodel.json" % (self._model_path), "w") as json_file:
            json_file.write(model_json)
        try:
            # serialize weights to HDF5
            self._model.save_weights(
                "%sweights.%05d.hdf5" % (self._model_path, self._epoch)
            )
            if self._verbose:
                print("Saved model to disk")
        except RuntimeError:
            if self._verbose:
                print("Couldn't save model to disk")

    def load_model(self, epoch=-1, model_path=None):
        if epoch < 0:
            epoch = self._epoch
        if model_path is None:
            model_path = self._model_path

        # load json and create model
        with open("%smodel.json" % (model_path), "r") as json_file:
            self._model = model_from_json(json_file.read())

        # load weights into new model
        self._model.load_weights("%sweights.%05d.hdf5" % (model_path, epoch))
        self._epoch = epoch
        self._model_path = model_path

        if self._verbose:
            print("Loaded model from disk")

    def fit(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)

    def predict_on_batch(self, *args, **kwargs):
        return self._model.predict_on_batch(*args, **kwargs)

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @property
    def config_path(self):
        return self._model_path + "config.json"

    @staticmethod
    def _set_precision(calculation_dtype, calculation_epsilon):
        # enable single/half/double precision
        K.set_floatx(calculation_dtype)
        K.set_epsilon(calculation_epsilon)

        # enable mixed precission
        if "float16" in calculation_dtype:

            mixed_precision.set_global_policy("mixed_float16")

    def _create_model(self):
        # initial definition of the sequential model
        self._model = keras.Sequential(name="open-neural-apc")
        self._model.add(
            InputLayer(
                input_shape=[None, *self._model_parameter["input_dimensions"]],
                dtype=self._training_parameter["calculation_dtype"],
            )
        )

    # input layer is just a dense layer
    # therefore we have to flatten the input frames
    def _add_input(self):
        self._model.add(
            Reshape(
                target_shape=(
                    -1,
                    np.multiply(*self._model_parameter["input_dimensions"]),
                ),
                name="InputReshape",
            )
        )
        self._model.add(Dense(self._model_parameter["lstm_width"], name="InputLayer"))
        self._model.add(
            Dropout(self._training_parameter["dropout_rate"], name="InputDropoutLayer")
        )
        self._model.add(LeakyReLU(name="InputActivation"))

    # the core network based on lstm
    def _add_core(self):
        # Doing this on purpose. The old LSTM implementation was faster than the new one
        v1rnn = self._training_parameter.get("v1RNN", False)

        for idx in range(self._model_parameter["lstm_depth"]):
            if v1rnn:
                lstm = CuDNNLSTM(
                    units=self._model_parameter["lstm_width"],
                    return_sequences=True,
                    name="CoreLayer{idx}" % idx,
                )
            else:
                lstm = LSTM(
                    units=self._model_parameter["lstm_width"],
                    return_sequences=True,
                    dropout=self._training_parameter["dropout_rate"],
                    name="CoreLayer%d" % idx,
                )

            if "bidirectional" in self._model_parameter:
                merge_mode = "concat"
                if "merge_mode" in self._model_parameter:
                    merge_mode = self._model_parameter["merge_mode"]
                if self._model_parameter["bidirectional"]:
                    lstm = Bidirectional(lstm, merge_mode=merge_mode)
            self._model.add(lstm)

    # the output layer just reducing the dimensionality to the regression output
    def _add_output(self):
        self._model.add(
            Dense(
                self._model_parameter["output_dimensions"],
                use_bias=True,
                name="OutputLayer",
            )
        )
        self._model.add(LeakyReLU(-1, name="OutputActivation"))

    def _add_callbacks(self):

        self._callbacks += [IncreaseEpochCustom(self)]
        self._callbacks += [
            SaveEveryNthEpochCustom(self, self._training_parameter["safe_steps"])
        ]
        # tensorboard callback
        self._callbacks += [
            tf.keras.callbacks.TensorBoard(
                log_dir=self._model_path + "/logs",
                histogram_freq=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch",
                profile_batch=0,
                embeddings_freq=0,
                embeddings_metadata=None,
            )
        ]

    def loss(self, y_true, y_pred):
        y_pred = K.cast(y_pred, dtype=K.floatx())

        output_dimensions = tf.shape(y_pred)[2]
        upper_bound = K.cast(y_true[:, :, :output_dimensions], dtype=K.floatx())
        lower_bound = K.cast(
            y_true[:, :, output_dimensions : 2 * output_dimensions], dtype=K.floatx()
        )

        return K.mean(
            self._calc_loss(upper_bound, lower_bound, y_pred), axis=0, keepdims=True
        )

    def accuracy(self, y_true, y_pred):
        y_pred = K.cast(y_pred, dtype=K.floatx())

        output_dimensions = tf.shape(y_pred)[2]
        upper_bound = K.cast(y_true[:, :, :output_dimensions], dtype=K.floatx())

        mask = K.cast(K.greater_equal(upper_bound, self._tf_zero), dtype=K.floatx())
        accuracy_mask = K.cast(y_true[:, :, 2 * output_dimensions :], dtype=K.floatx())

        # because the accuracy_mask is originally also padded with -1, we mask it
        accuracy_mask = mask * accuracy_mask
        error_with_slack = K.abs(y_pred - upper_bound) - self._slack
        error_with_slack = K.cast(
            K.less_equal(error_with_slack, self._tf_zero), dtype=K.floatx()
        )
        # number of right predicted sequences divided by count of sequences
        return K.sum(accuracy_mask * error_with_slack) / K.sum(accuracy_mask)

    @staticmethod
    def _aux_losses(mask, prediction, tf_zero, tf_one):
        # try to predict close to integer values
        # (error = distance to closest integer)
        integer_error = mask * (prediction - K.round(prediction))

        # try to not change prediction too much through time
        # (error = distance/change to previous prediction)
        small_zero = K.zeros_like(prediction[:, :1, :])
        stabelize_change_forward = K.maximum(
            tf_zero,
            prediction - K.concatenate([prediction[:, 1:, :], small_zero], axis=1),
        )
        stabelize_change_backward = mask * K.concatenate(
            [small_zero, stabelize_change_forward[:, :-1, :]], axis=1
        )

        # hold prediction when input is a constant -1 filled image (padding)
        # (error = distance/change to last valid prediction)
        inverted_mask = K.cast(K.less(mask, tf_one), dtype=K.floatx())
        very_last_valid_frame = mask * K.concatenate(
            [inverted_mask[:, 1:, :], small_zero], axis=1
        )
        last_prediction = K.sum(
            prediction * very_last_valid_frame, axis=1, keepdims=True
        )
        freezed_last_prediction = inverted_mask * (prediction - last_prediction)

        # combine all auxilary losses
        return K.concatenate(
            [
                K.abs(integer_error),
                K.abs(stabelize_change_backward),
                K.abs(freezed_last_prediction),
            ],
            axis=0,
        )

    def _calc_loss(self, upper_bound, lower_bound, prediction):
        mask = K.cast(K.greater_equal(upper_bound, self._tf_zero), dtype=K.floatx())
        # main error to the label (the predictions outside the bounding boxes)
        error = mask * (
            K.maximum(self._tf_zero, prediction - upper_bound)
            + K.minimum(self._tf_zero, prediction - lower_bound)
        )
        # additional losses (independent from label)
        aux_loss = self._aux_losses(mask, prediction, self._tf_zero, self._tf_one)
        return K.abs(error) + K.mean(aux_loss, axis=0, keepdims=True) / self._aux_scale
