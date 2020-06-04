import numpy as np
from tensorflow import keras as keras


class Preprocess():
    def __init__(self,sequence_list,labels_list,input_scaling_factor,training_parameter,calculation_dtype):
        self.permute = np.random.randint(0,2,(len(sequence_list),2))
        self.labels_list = labels_list
        self.sequence_list = sequence_list
        self.input_scaling_factor = input_scaling_factor
        self.training_parameter = training_parameter
        self.calculation_dtype = calculation_dtype

    # prepare epoch
    def prepareIndices(self):
        indices = np.arange(len(self.sequence_list))
        if self.training:
            np.random.shuffle(indices)
        batches = []
        while(True):
            samples = []
            for _ in range(self.training_parameter['batch size']):
                num = np.random.randint(self.training_parameter['minimum concatenation'],\
                                          self.training_parameter['maximum concatenation']+1)
                num = min(num,np.shape(indices)[0])
                choice = indices[:num]
                indices = indices[num:]
                if np.shape(choice)[0] > 0:
                    samples += [choice]
            batches += [samples]
            if np.shape(indices)[0] == 0:
                break
        return batches

    def epochWrapper(self,indices,function):
        def batchWrapper(batch,function):
            batches = []
            for sample in batch:
                batches += [function(sample)]
            return batches
        
        epoch = []
        for batch in indices:
            epoch += [batchWrapper(batch,function)]
        return epoch


    def sequenceEpoch(self,indices):
        one_iter = []
        for idx in indices:
            sequence = np.array(self.sequence_list[idx])
            sequence = sequence/self.input_scaling_factor

            permutations = self.permute[idx]
            if permutations[0]:
                # mirror through time back/forth
                sequence = sequence[::-1,:,:]
            if permutations[1]:
                # mirror left/right
                sequence = sequence[:,:,::-1]

            one_iter += [sequence]
        return np.concatenate(one_iter)


    def labelEpoch(self,indices):
        upper_bound = []
        lower_bound = []

        in_ = []
        out_ = []

        for idx in indices:
            in_label,out_label = self.labels_list[idx]
            length = np.shape(self.sequence_list[idx])[0]
            ones = np.ones((length,1))

            in_ += [int(in_label)]
            out_ += [int(out_label)]

            permutations = self.permute[idx]
            if permutations[0]:
                in_[-1] = int(out_label)
                out_[-1] = int(in_label)

            up_in = ones*sum(in_)
            up_out = ones*sum(out_)
            upper_bound += [np.concatenate([up_in,up_out],axis=1)] 

            low_in = ones*sum(in_[:-1])
            low_out = ones*sum(out_[:-1])
            low_in[-1] = up_in[-1]
            low_out[-1] = up_out[-1]       
            lower_bound += [np.concatenate([low_in,low_out],axis=1)]

        upper_bound = np.concatenate(upper_bound)
        lower_bound = np.concatenate(lower_bound)
        return np.concatenate([upper_bound,lower_bound],axis=1)

    def accuracyEpoch(self,indices):
        hit_mask = []
        for idx in indices:
            sequence = self.sequence_list[idx]
            hit_mask += [np.zeros((np.shape(sequence)[0],1))]
            hit_mask[-1][-1] = 1
        return np.concatenate(hit_mask)

    def prepareEpoch(self,training=True):
        self.training = training
        indices = self.prepareIndices()
        # reverse left/right and forward/backward only during training (every epoch is a new permutation)
        if training:
            self.permute[:] = np.random.randint(0,2,(len(self.sequence_list),2))
        else:
            self.permute[:] = np.zeros((len(self.sequence_list),2))
        
        def padEpoch(array):
            for idx,elem in enumerate(array):
                array[idx] = keras.preprocessing.sequence.pad_sequences(elem, maxlen=None,\
                                    dtype=self.calculation_dtype, padding='post', value=-1.0)
            return array
        
        video_sequences =  padEpoch(self.epochWrapper(indices,self.sequenceEpoch))
        label_mask =  padEpoch(self.epochWrapper(indices,self.labelEpoch))
        accuracy_mask = padEpoch(self.epochWrapper(indices,self.accuracyEpoch))

        label = []
        for idx,elem in enumerate(label_mask):
            label += [np.concatenate([elem, accuracy_mask[idx]],axis=-1)]

        return video_sequences,label