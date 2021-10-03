# -*- coding: utf-8 -*-
# Copyright (c) 2020-2021, Nico Jahn
# All rights reserved.
# pylint: disable=missing-module-docstring
import json

import tensorflow as tf

# load model config
def load_config(config_path="config.json", verbose=1):

    with open(config_path, "r") as config_file:
        config_data = json.load(config_file)

    # extract parameter classes
    data_parameter = config_data["data_parameter"]
    model_parameter = config_data["model_parameter"]
    training_parameter = config_data["training_parameter"]

    # show content of config
    if verbose:
        print(json.dumps(config_data, indent=2, sort_keys=True))

    return (data_parameter, model_parameter, training_parameter)


def allow_growth():

    # Copied from: https://tensorflow.google.cn/guide/gpu?hl=en#limiting_gpu_memory_growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:  # pylint: disable=invalid-name
            print(e)
