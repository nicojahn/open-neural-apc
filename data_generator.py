# Copyright (c) 2020, Nico Jahn
# All rights reserved.

import numpy as np
from tensorflow import keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self,data,training_parameter):
        self.num_sequences = len(data)
        self.permute = np.random.randint(0,2,(self.num_sequences,2))
        self.data = data

        self.training_parameter = training_parameter
        self.calculation_dtype = training_parameter["calculation_dtype"]

        self.concat_length = self.training_parameter['concatenation_length']
        self.batch_size = self.training_parameter['batch_size']
        self.num_batches = np.ceil((float(self.num_sequences)/self.concat_length)/self.batch_size).astype(np.int32)

    def prepareIndices(self):
        indices = np.arange(self.num_sequences)
        if self.training:
            np.random.shuffle(indices)
        # to handle the indices easier, i'm going to pad with -1 until we can reshape the indices properly and discard the -1 indices later
        max_sequences = (self.num_batches*self.batch_size*self.concat_length)
        indices = np.append(indices, -1*np.ones(max_sequences-self.num_sequences))
        batches = indices.reshape((self.num_batches,self.batch_size,self.concat_length))
        return batches.astype(np.int32)

    def videoSample(self,idx):
        sequence = np.asarray(self.data[idx],dtype=self.calculation_dtype)
        # mirror through time back/forth and mirror left/right
        return sequence[::self.permute[idx][0],:,::self.permute[idx][1]]

    def labelSample(self,idx):
        labels = np.asarray(self.data.getLabel(idx)[::self.permute[idx][0]],dtype=self.calculation_dtype)
        length = self.data.getLength(idx)
        bound = np.zeros((length,2*self.data.getNumClasses()),dtype=self.calculation_dtype)
        # the upper bound (always equals labels) is already valid on the first frame
        # the lower bound is mostly 0, but for the last frame it's also the label
        # i'm only setting one element, since when concatenated, we have to calculate the cumsum over the labels 
        bound[0,:self.data.getNumClasses()] = labels
        bound[-1,self.data.getNumClasses():] = labels
        return bound

    def accuracySample(self,idx):
        # the accuracy can only be determined on the last frame
        acc = np.zeros((self.data.getLength(idx),1),dtype=self.calculation_dtype)
        acc[-1] = 1
        return acc

    def padBatch(self,seq):
        # padding/masking the sequences with -1 to the longest sequence in the batch
        return pad_sequences(seq, maxlen=None, dtype=self.calculation_dtype, padding='post', value=-1.0)

    def combineMasksBatch(self,l_mask,a_mask):
        # stacks the label mask (batch_size x sequence_length x 2*num_classes) and the accuracy mask (batch_size x sequence_length x 1)
        return np.dstack([l_mask,a_mask])

    def batchWrapper(self,batch,function):
        batches = []
        # operating on a single batch element
        for indices in batch:
            result = []
            # operating on the sequences of a batch element
            for idx in indices:
                if idx < 0: continue
                # prepares exacly 1 sequence
                result += [function(idx)]
            if len(result):
                # concatenates the sequences to 1 batch element
                batches += [np.concatenate(result)]
        return batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        batch = self.indices[idx]
        video_sequences = self.padBatch(self.batchWrapper(batch,self.videoSample))

        # np.cumsum(...,axis=1) without removing the -1 padding values
        label_mask = self.padBatch(self.batchWrapper(batch,self.labelSample))
        indices = np.where(label_mask==-1)
        label_mask = np.cumsum(label_mask,axis=1)
        label_mask[indices] = -1.

        accuracy_mask = self.padBatch(self.batchWrapper(batch,self.accuracySample))
        return video_sequences, self.combineMasksBatch(label_mask,accuracy_mask)

    def on_epoch_end(self,training=True):
        self.training = training
        self.indices = self.prepareIndices()

        # reverse left/right and forward/backward only during training (every epoch is a new permutation)
        possible_permutations = 0
        if training:
            possible_permutations = 1
        # this is the 'step' of the arrays (either 1 for no permutations or -1 for reversing a sequence)
        self.permute = np.int32(1 - (2 * np.random.randint(0,possible_permutations+1,(self.num_sequences,2))))