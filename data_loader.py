# -*- coding: utf-8 -*-
# Copyright (c) 2020-2021, Nico Jahn
# All rights reserved.
# pylint: disable=missing-module-docstring
import os

import h5py
import numpy as np
import pandas as pd

from utils import download
from utils import load_config

# the official TU Berlin DepositOnce reference of version 1 of the data set
deposit_once = {
    "data": {
        "url": "https://www.depositonce.tu-berlin.de/bitstream/11303/13421/3/berlin-apc.h5",
        "file_path": "data/berlin-apc.h5",
    },
    "labels": {
        "url": "https://www.depositonce.tu-berlin.de/bitstream/11303/13421/2/berlin-apc.csv",
        "file_path": "data/berlin-apc.csv",
    },
    "readme": {
        "url": "https://www.depositonce.tu-berlin.de/bitstream/11303/13421/6/readme.pdf",
        "file_path": "data/readme.pdf",
    },
}

# github.com/nicojahn free AWS S3 mirror for faster access (version 1 of the data set)
aws_mirror = {
    "data": {
        "url": "https://berlin-apc.s3.eu-central-1.amazonaws.com/berlin-apc.h5",
        "file_path": "data/berlin-apc.h5",
    },
    "labels": {
        "url": "https://berlin-apc.s3.eu-central-1.amazonaws.com/berlin-apc.csv",
        "file_path": "data/berlin-apc.csv",
    },
    "readme": {
        "url": "https://berlin-apc.s3.eu-central-1.amazonaws.com/readme.pdf",
        "file_path": "data/readme.pdf",
    },
}


class DataLoader:
    def __init__(self, training_parameter, data_fname="data/data.h5", mode="training"):
        self._frame_stride = training_parameter["frame_stride"]

        self._sequences = _mmap_h5(data_fname, f"/{mode}/sequences")
        self._labels = _mmap_h5(data_fname, f"/{mode}/labels")
        self._lengths = _mmap_h5(data_fname, f"/{mode}/lengths")
        self._file_names = _mmap_h5(data_fname, f"/{mode}/file_names")

        self._num_classes = np.shape(self._labels)[1]

    def __getitem__(self, idx):
        offset = np.sum(self._lengths[:idx])
        sequence_length = self._lengths[idx]
        return self._sequences[offset : offset + sequence_length : self._frame_stride]

    def get_label(self, idx):
        return self._labels[idx]

    def get_length(self, idx):
        return np.ceil(self._lengths[idx] / self._frame_stride).astype(np.int32)

    def get_file_name(self, idx):
        return self._file_names[idx]

    @property
    def num_classes(self):
        return self._num_classes

    def __len__(self):
        return self._lengths.shape[0]


# Copied from Cyrille Rossants HDF5 benchmark: https://gist.github.com/rossant/7b4704e8caeb8f173084
def _mmap_h5(path, h5path):
    with h5py.File(path, "r") as h5file:
        # check if the h5path exists
        assert h5path in h5file.keys()
        dataset = h5file[h5path]
        # We get the dataset address in the HDF5 file.
        offset = dataset.id.get_offset()
        dtype = dataset.dtype
        shape = dataset.shape
        # return None if dataset is empty (offset is None)
        if offset is None:
            return np.ndarray(shape, dtype=dtype)
        # We ensure we have a non-compressed contiguous array.
        assert dataset.chunks is None
        assert dataset.compression is None
        assert offset > 0
    arr = np.memmap(path, mode="r", shape=shape, offset=offset, dtype=dtype)
    return arr


# pylint: disable=missing-module-docstring
class BerlinAPCHelper:
    def __init__(self, config, door_at_bottom=True):
        self._config = config
        self._data_parameter, _, _ = load_config(verbose=0)
        self._door_at_bottom = door_at_bottom
        self._new_h5_data = {"training": {}}
        self._old_h5_data = {"validation": {}}

    def download(self):
        if not os.path.exists(self._config["data"]["file_path"]):
            download(self._config["data"]["url"], self._config["data"]["file_path"])

        if not os.path.exists(self._config["labels"]["file_path"]):
            download(self._config["labels"]["url"], self._config["labels"]["file_path"])

        if not os.path.exists(self._config["readme"]["file_path"]):
            download(self._config["readme"]["url"], self._config["readme"]["file_path"])

    def _load_new_data(self):

        # Load HDF5 and CSV file as described in the "readme.pdf" from the dataset
        dataset = h5py.File(self._config["data"]["file_path"], mode="r")
        df_labels = pd.read_csv(
            self._config["labels"]["file_path"],
            names=["sequence_name", "n_boarding", "n_alighting"],
        )

        sequences = list()
        lengths = list()
        for _, value in dataset.items():
            sequences.append(value)
            lengths.append(value.shape[0])

        self._new_h5_data["training"]["sequences"] = np.concatenate(sequences)
        self._new_h5_data["training"]["labels"] = df_labels[
            ["n_boarding", "n_alighting"]
        ].to_numpy()
        self._new_h5_data["training"]["lengths"] = np.asarray(lengths)
        self._new_h5_data["training"]["file_names"] = df_labels[
            "sequence_name"
        ].to_numpy()

        # Flipping the image upside down, if door should be at the ***bottom***
        if self._door_at_bottom:
            self._new_h5_data["training"]["sequences"] = self._new_h5_data["training"][
                "sequences"
            ][:, ::-1]

    def _load_old_data(self):
        validation_data = DataLoader(
            {"frame_stride": 1}, self._data_parameter["data"], "validation"
        )

        self._old_h5_data["validation"]["sequences"] = np.concatenate(
            list(iter(validation_data))
        )
        self._old_h5_data["validation"]["labels"] = np.concatenate(
            [[validation_data.get_label(idx) for idx, _ in enumerate(validation_data)]]
        )
        self._old_h5_data["validation"]["lengths"] = np.concatenate(
            [[elem.shape[0] for elem in iter(validation_data)]]
        )
        self._old_h5_data["validation"]["file_names"] = np.concatenate(
            [[f"open_neural_apc{idx:X}" for idx, _ in enumerate(validation_data)]]
        )

        # Flipping the image upside down, if door should be at the ***top***
        if not self._door_at_bottom:
            self._old_h5_data["validation"]["sequences"] = self._old_h5_data[
                "validation"
            ]["sequences"][:, ::-1]

        del validation_data

    def convert(self):

        self._load_old_data()
        self._load_new_data()

        with h5py.File(self._data_parameter["data"], "w") as h5file:

            train_g = h5file.create_group("training")
            train_g.create_dataset(
                "sequences",
                data=self._new_h5_data["training"]["sequences"].astype(np.float16),
            )
            train_g.create_dataset(
                "labels", data=self._new_h5_data["training"]["labels"].astype(np.int32)
            )
            train_g.create_dataset(
                "lengths",
                data=self._new_h5_data["training"]["lengths"].astype(np.uint32),
            )
            train_g.create_dataset(
                "file_names",
                data=self._new_h5_data["training"]["file_names"].astype(
                    h5py.string_dtype(length=16)
                ),
            )

            valid_g = h5file.create_group("validation")
            valid_g.create_dataset(
                "sequences", data=self._old_h5_data["validation"]["sequences"]
            )
            valid_g.create_dataset(
                "labels", data=self._old_h5_data["validation"]["labels"]
            )
            valid_g.create_dataset(
                "lengths", data=self._old_h5_data["validation"]["lengths"]
            )
            valid_g.create_dataset(
                "file_names",
                data=self._old_h5_data["validation"]["file_names"].astype(
                    h5py.string_dtype(length=16)
                ),
            )


def main():
    berlin_apc_helper = BerlinAPCHelper(aws_mirror)
    berlin_apc_helper.download()
    berlin_apc_helper.convert()


if __name__ == "__main__":
    main()
