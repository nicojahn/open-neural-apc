import tensorflow as tf
import numpy as np
import csv
from data_loader import loader

x_height = 20
x_width = 25
output_dim = 2

minimum_concat = 2
maximum_concat = 5

def prepareData(x,y, batch_size):
    # augment data (mirror along allowed axis)
    augmentations(x,y)
    # shuffles the indices (therefore, concatenation is every batch/epoch differently)
    indices = shuffleAllSequences(x)
    # concatenate sequences in the range of [minimum_concat,maximum_concat]
    concatenated_x, concatenated_y, indices_sampling = concatToLongerSequences(x,y,indices)
    # creates batches from the concatenated sequenes
    batch_x = createBatches(concatenated_x,batch_size)
    # add padding to the x sequences
    batch_x = list(map(lambda elem: addPadding(elem),batch_x))
    return batch_x, None, indices_sampling, y

### has to be done on the lists (individual sequences, not concatenated)
def augmentations(x,y):
    #batch_size, possibilities
    for idx,z in enumerate(x):
        augmentations = np.random.randint(0,2,[2])
        reshaped_x = np.reshape(np.asarray(z),[-1,x_height,x_width])
        if augmentations[0] == 1:
            # mirror along vertical axis
            x[idx] = np.reshape(np.flip(reshaped_x,2),[-1,x_height*x_width])
        if augmentations[1] == 1:
            # reverse sequence (and the label)
            x[idx] = np.reshape(np.flip(reshaped_x,0),[-1,x_height*x_width])
            y[idx] = np.flip(y[idx])

# partition data into batch_size large buckets
def createBatches(data, batch_size):
    batches = np.arange(0,len(data),batch_size)
    batch_data = [data[idx:idx+batch_size] for idx in batches]
    return batch_data

# To be fed into tensorflow, the sequences within a batch have to have the same length
# Therefore, i add a padding, which also functions later as binary mask for the error function
def addPadding(seq,max_length=-1):
    input_dims = np.shape(np.asarray(seq[0]))[1]
    if max_length<0:
        #add Padding
        max_length = max(getSequenceLengths(seq))
    
    # with the padding of -1 i can later detect when a sequence has ended
    def padOneSequence(seq):
        pad_length = (max_length-np.shape(seq)[0])
        return np.pad(seq,((0,pad_length),(0,0)), constant_values=(-1.))
    
    padded = np.asarray(list(map(padOneSequence,seq)))
    return padded

def concatToLongerSequences(x,y, indices):
    # lists for collecting all the sequences, labels and indices
    concatenated_x = []
    concatenated_y = []
    indices_sampling = []
    
    # as long as there are leftover indices
    #   take 'concat_length' sequences/labels and concatenate respectively
    while np.shape(indices)[0] > 0:
        # at least 'minimum' sequences for concatenation
        minimum = min(minimum_concat,np.shape(indices)[0])
        # at most 'maximum' sequences for concatenation
        maximum = min(maximum_concat,np.shape(indices)[0])
        # the concatenantion length for one element in one batch
        concat_length = np.random.random_integers(minimum,maximum)
        
        # obtain indices and remove them from the indices pool
        choice = indices[:concat_length]
        indices = np.setdiff1d(indices,choice)
        
        # concatenate the sequences to a longer one
        tmp_x = np.concatenate([x[idx] for idx in choice],axis=0)
        # do the same for the labels
        tmp_y = np.concatenate([y[idx] for idx in choice])
        
        # collect them together with other concatenated sequences
        # to later remember which sequences (with which length) where chosen
        #       save the indices (choice) as well
        concatenated_x += [np.asarray(tmp_x,dtype=float)]
        concatenated_y += [np.asarray(tmp_y,dtype=float)]
        indices_sampling += [choice]
    return concatenated_x, concatenated_y, indices_sampling

# getting x: shape = (batchsize,seqence length, input dims)
def shuffleAllSequences(x):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    return indices

#returns a list of the lenght of the inner lists
def getSequenceLengths(input_list):
    return list(map(lambda x: len(x),input_list))

# read data from files and choose keep_noise-mode:
#   keep_noise==True: read with noise
#   keep_noise==False: read without noise
# frame_hops defines the rate of pictures loaded:
#   1: every picture
#   n: every n-th picture
def readData(label_directory, label_file, csv_directory, keep_noise=False, frame_hops=1):
    def readMetainfos(input_file):
        with open('%s'%(input_file), newline='') as f:
            return list(csv.reader(f, delimiter=','))
    # loads just the metainformation from a file (filename,in,out,seq_length)
    metaInfos = readMetainfos('%s%s'%(label_directory,label_file))
    data, label = loader(csv_directory, frame_hops).load_file_from_list(metaInfos)

#    # in contrast to my bachelors thesis i do not remove the noise in this case
#    # Since every frame in every sequence has to be analyzed individually it not practical for usage of other datasets
#    # filter noise
#    if not keep_noise:
#        for index,row in enumerate(metaInfos[:restricted_index]):
#            _, _, _, _, _, _, noise = row
#            sequence = np.ones((np.shape(data[index])[0]))
#
#            for entry in noise.split(';'):
#                if len(entry) > 0:
#                    start, end = np.asarray(entry.split(':'),dtype=int)
#                    indices = np.arange(start/frame_hops,(end/frame_hops),dtype=int)
#                    sequence[indices] = 0
#        data[index] = np.array(data[index])[sequence==1]

    return data, label


# creates my important mask and the padded label
# label_mask: shape=(batch_size,seq_length,output_dim), is an array of zeros where only the indices where a sequence ends is 1 (binary mask)
# bound_mask: shape=(batch_size,seq_length,output_dim), for concatenation necessary (is 1 for every fram of first sequence, 2 for every frame of second sequence, ...). Can then later be used to create accumulated labels/bounds
def createMasks(indices_sampling, original_length_list, original_lables_list, batch_size):
    
    original_lables = [[original_lables_list[idx] for idx in batch] for batch in indices_sampling]
    original_length = [[original_length_list[idx] for idx in batch] for batch in indices_sampling]
    labels_batch = createBatches(original_lables,batch_size)
    length_batch = createBatches(original_length,batch_size)
    labels_batch = list(map(lambda elem: addPadding(elem,maximum_concat),labels_batch))

    label_mask = []
    bound_mask = []
    for seq_lengths in length_batch:
        length = max([sum(seq) for seq in seq_lengths])
        label_mask_ = np.zeros((len(seq_lengths),length,output_dim))
        bound_mask_ = np.ones((len(seq_lengths),length,output_dim))
        for batch_idx,elem in enumerate(seq_lengths):
            for seq_idx, _ in enumerate(elem):
                label_mask_[batch_idx,sum(elem[:seq_idx+1])-1] = 1
                bound_mask_[batch_idx,sum(elem[:seq_idx+1]):] += 1
        label_mask += [label_mask_]
        bound_mask += [bound_mask_]

    return label_mask, bound_mask, labels_batch
