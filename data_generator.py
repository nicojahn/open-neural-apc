# -*- coding: utf-8 -*-
# Copyright (c) 2020-2021, Nico Jahn
# All rights reserved.
# pylint: disable=missing-module-docstring, no-name-in-module
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, data, training_parameter, training=False):
        self._num_sequences = len(data)
        self._permute = np.random.randint(0, 2, (self._num_sequences, 2))
        self._data = data
        self._training = training

        self._calculation_dtype = training_parameter["calculation_dtype"]
        self._concat_length = training_parameter["concatenation_length"]
        self._batch_size = training_parameter["batch_size"]

        self._num_batches = np.ceil(
            self._num_sequences / (self._concat_length * self._batch_size)
        ).astype(np.int32)

        self._indices = np.arange(self._num_sequences)
        self.on_epoch_end()

    def _prepare_indices(self):
        indices = np.arange(self._num_sequences)
        if self._training:
            np.random.shuffle(indices)
        # to handle the indices easier, i'm going to pad with -1
        # so we can reshape the indices properly and discard the -1 indices later
        max_sequences = self._num_batches * self._batch_size * self._concat_length
        indices_padding = -1 * np.ones(max_sequences - self._num_sequences)
        indices = np.append(indices, indices_padding)
        batches = indices.reshape(
            (self._num_batches, self._batch_size, self._concat_length)
        )
        return batches.astype(np.int32)

    def _video_sample(self, idx):
        sequence = np.asarray(self._data[idx], dtype=self._calculation_dtype)
        # mirror through time back/forth and mirror left/right
        return sequence[:: self._permute[idx][0], :, :: self._permute[idx][1]]

    def _label_sample(self, idx):
        labels = np.asarray(
            self._data.get_label(idx)[:: self._permute[idx][0]],
            dtype=self._calculation_dtype,
        )
        length = self._data.get_length(idx)
        bound = np.zeros(
            (length, 2 * self._data.num_classes), dtype=self._calculation_dtype
        )
        # the upper bound (always equals labels) is already valid on the first frame
        # the lower bound is mostly 0, but for the last frame it's also the label
        # i'm only setting one element as we calculate the cumsum over the labels later on
        bound[0, : self._data.num_classes] = labels
        bound[-1, self._data.num_classes :] = labels
        return bound

    def _accuracy_sample(self, idx):
        # the accuracy can only be determined on the last frame
        acc = np.zeros((self._data.get_length(idx), 1), dtype=self._calculation_dtype)
        acc[-1] = 1
        return acc

    def _pad_batch(self, seq):
        # padding/masking the sequences with -1 to the longest sequence in the batch
        return pad_sequences(
            seq, maxlen=None, dtype=self._calculation_dtype, padding="post", value=-1.0
        )

    @staticmethod
    def _combine_masks_batch(l_mask, a_mask):
        # stacks the label mask (batch_size x sequence_length x 2*num_classes)
        # and the accuracy mask (batch_size x sequence_length x 1)
        return np.dstack([l_mask, a_mask])

    @staticmethod
    def _batch_wrapper(batch, function):
        batches = []
        # operating on a single batch element
        for indices in batch:
            result = []
            # operating on the sequences of a batch element
            for idx in indices:
                if idx < 0:
                    continue
                # prepares exacly 1 sequence
                result += [function(idx)]
            if len(result) > 0:
                # concatenates the sequences to 1 batch element
                batches += [np.concatenate(result)]
        return batches

    def __len__(self):
        return self._num_batches

    def __getitem__(self, idx):
        batch = self._indices[idx]
        video_sequences = self._pad_batch(
            self._batch_wrapper(batch, self._video_sample)
        )

        # np.cumsum(...,axis=1) without removing the -1 padding values
        label_mask = self._pad_batch(self._batch_wrapper(batch, self._label_sample))
        indices = np.where(label_mask == -1)
        label_mask = np.cumsum(label_mask, axis=1)
        label_mask[indices] = -1.0

        accuracy_mask = self._pad_batch(
            self._batch_wrapper(batch, self._accuracy_sample)
        )
        return (video_sequences, self._combine_masks_batch(label_mask, accuracy_mask))

    def on_epoch_end(self):
        self._indices = self._prepare_indices()

        # reverse left/right and forward/backward only during training
        # (every epoch is a new permutation)
        possible_permutations = 0
        if self._training:
            possible_permutations = 1
        # this is calculation of the 'step' of the array indices
        # (either 1 for no permutations or -1 for reversing a sequence)
        self._permute = 1 - 2 * np.int32(
            (np.random.randint(0, possible_permutations + 1, (self._num_sequences, 2)))
        )
