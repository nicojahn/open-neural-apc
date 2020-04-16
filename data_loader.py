def readData(model_parameter,training_parameter,data_fname,data_length_fname,sequence_dtype,labels_dtype):
    import time as t
    start = t.time()
    print("Started reading files: %s"%t.strftime('%H:%M:%S %Y-%m-%d', t.localtime(start)))

    import numpy as np
    fp_lengths = np.memmap(data_length_fname, dtype=labels_dtype, mode='r')
    fp_lengths = np.reshape(fp_lengths,[-1,1+model_parameter['output dimensions']])
    
    labels = fp_lengths[:,1:]
    fp_lengths = fp_lengths[:,0]
    
    sequences = np.memmap(data_fname, dtype=sequence_dtype, mode='r')
    sequences = np.reshape(sequences,[-1,*model_parameter['input dimensions']])
    
    
    jump_frames = training_parameter['jump input frames']
    
    sequence_list, labels_list = [], []
    for idx,length in enumerate(fp_lengths):
        
        offset = sum(fp_lengths[:idx])
        sequence_length = fp_lengths[idx]
        
        sequence = sequences[offset:offset+sequence_length:jump_frames]
        
        sequence_list += [sequence]
        labels_list += [labels[idx]]
    
    num_sequences = len(sequence_list)
    elapsed_time = round(t.time()-start,2)
    print( "Finished reading %d sequences. Took %f seconds." % (num_sequences,elapsed_time))
    return sequence_list, labels_list