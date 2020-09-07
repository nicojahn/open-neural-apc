import numpy as np
from tensorflow import keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self,sequence_list,labels_list,input_scaling_factor,training_parameter,calculation_dtype):
        self.permute = np.random.randint(0,2,(len(sequence_list),2))
        self.labels_list = labels_list
        self.sequence_list = sequence_list
        self.input_scaling_factor = input_scaling_factor
        self.training_parameter = training_parameter
        self.calculation_dtype = calculation_dtype
        self.concat_length = self.training_parameter['minimum concatenation']
        self.batch_size = self.training_parameter['batch size']
        self.num_batches = np.ceil((float(len(sequence_list))/self.concat_length)/self.batch_size).astype(int)

    def prepareIndices(self):
        indices = np.arange(len(self.sequence_list))
        if self.training:
            np.random.shuffle(indices)
        max_sequences = (self.num_batches*self.batch_size*self.concat_length)        
        indices = np.append(indices, -1*np.ones(max_sequences-len(self.sequence_list)))
        batches = indices.reshape((self.num_batches,self.batch_size,self.concat_length))
        return batches.astype(np.int32)

    def videoSample(self,idx):
        sequence = np.array(self.sequence_list[idx],dtype=np.float16)
        sequence = sequence/self.input_scaling_factor

        permutations = self.permute[idx]
        if permutations[0]:
            # mirror through time back/forth
            sequence = sequence[::-1,:,:]
        if permutations[1]:
            # mirror left/right
            sequence = sequence[:,:,::-1]
        return sequence

    def labelSample(self,idx):
        length = np.shape(self.sequence_list[idx])[0]
        permutations = self.permute[idx]
        labels = np.array(self.labels_list[idx],dtype=np.int32)
        if permutations[0]:
            labels[:] = labels[::-1]
        bound = np.zeros((length,4),dtype=np.int32)
        bound[0,:2] = labels
        bound[-1,2:] = labels
        return bound
    
    def accuracySample(self,idx):
        acc = np.zeros((np.shape(self.sequence_list[idx])[0],1),dtype=np.int32)
        acc[-1] = 1
        return acc
    
    def padBatch(self,elem):
        return pad_sequences(elem, maxlen=None, dtype=self.calculation_dtype, padding='post', value=-1.0)
    
    def combineMasksBatch(self,l_mask,a_mask):
        return np.concatenate([l_mask,a_mask],axis=-1)
    
    def batchWrapper(self,batch,function):
        batches = []
        for indices in batch:
            result = []
            for idx in indices:
                if idx < 0: continue
                result += [function(idx)]
            if len(result):
                batches += [np.concatenate(result)]
        return batches
    
    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        batch = self.indices[idx]
        video_sequences = self.padBatch(self.batchWrapper(batch,self.videoSample))
        
        # np.cumsum(...,axis=1) without removing the -1 values
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
        if training:
            self.permute[:] = np.random.randint(0,2,(len(self.sequence_list),2))
        else:
            self.permute[:] = np.zeros((len(self.sequence_list),2))