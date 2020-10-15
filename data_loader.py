# Copyright (c) 2020, Nico Jahn
# All rights reserved.

import numpy as np

class DataLoader:
    def __init__(self,training_parameter,data_fname="data.h5",mode="training"):
        self.frame_stride = training_parameter['frame_stride']
        
        self.sequences = _mmap_h5(data_fname, f'/{mode}/sequences')
        self.labels = _mmap_h5(data_fname, f'/{mode}/labels')
        self.lengths = _mmap_h5(data_fname, f'/{mode}/lengths')
        
        self.num_classes = np.shape(self.labels)[1]

    def __getitem__(self,idx):
        offset = np.sum(self.lengths[:idx])
        sequence_length = self.lengths[idx]
        return self.sequences[offset:offset+sequence_length:self.frame_stride]
    
    def getLabel(self,idx):
        return self.labels[idx]
    
    def getLength(self,idx):
        return np.ceil(self.lengths[idx]/self.frame_stride).astype(np.int32)
    
    def getNumClasses(self):
        return self.num_classes
    
    def __len__(self):
        return self.lengths.shape[0]

# Copied from Cyrille Rossants HDF5 benchmark: https://gist.github.com/rossant/7b4704e8caeb8f173084
import h5py
def _mmap_h5(path, h5path):
    with h5py.File(path,'r') as f:
        # check if the h5path exists
        assert h5path in f.keys()
        ds = f[h5path]
        # We get the dataset address in the HDF5 file.
        offset = ds.id.get_offset()
        dtype = ds.dtype
        shape = ds.shape
        # return None if dataset is empty (offset is None)
        if offset == None:
            return np.ndarray(shape,dtype=dtype)
        # We ensure we have a non-compressed contiguous array.
        assert ds.chunks is None
        assert ds.compression is None
        assert offset > 0
    arr = np.memmap(path, mode='r', shape=shape, offset=offset, dtype=dtype)
    return arr