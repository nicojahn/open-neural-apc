# -*- coding: utf-8 -*-
# Copyright (c) 2020-2021, Nico Jahn
# All rights reserved.
# pylint: disable=missing-module-docstring
import json

import requests
import tensorflow as tf
from tqdm import tqdm

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


# all credits go to yanqd0 for the tqdm status bar:
# https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024
    ) as status_bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            status_bar.update(size)
