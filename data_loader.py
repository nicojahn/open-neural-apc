# Copyright (c) 2020, Nico Jahn
# All rights reserved.

import numpy as np

class DataLoader:
    def __init__(self,model_parameter,training_parameter,data_fname,data_length_fname,sequence_dtype,labels_dtype):
        self.jump_frames = training_parameter['jump_frames']

        sequences = np.memmap(data_fname, dtype=sequence_dtype, mode='r')
        self.sequences = np.reshape(sequences,[-1,*model_parameter['input_dimensions']])

        lengths_and_label = np.memmap(data_length_fname, dtype=labels_dtype, mode='r')
        self.lengths_and_label = np.reshape(lengths_and_label,[-1,1+model_parameter['output_dimensions']])

        self.lengths = self.lengths_and_label[:,0]
        self.labels = self.lengths_and_label[:,1:]
        
        self.frame_shape = model_parameter['input_dimensions']
        self.num_classes = model_parameter['output_dimensions']

    def __getitem__(self,idx):
        offset = sum(self.lengths[:idx])
        sequence_length = self.lengths[idx]
        return self.sequences[offset:offset+sequence_length:self.jump_frames]
    
    def getLabel(self,idx):
        return self.labels[idx]
    
    def getLength(self,idx):
        return np.ceil(self.lengths[idx]/self.jump_frames).astype(np.int32)
    
    def getNumClasses(self):
        return self.num_classes
    
    def numSequences(self):
        return self.lengths.shape[0]