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

### Helpers for converting the old CSV files into beautiful memory-mapped binaries
def getNamesOfAllDataFiles(path,filetype='csv'):
    from pathlib import Path
    train_files = Path(path).rglob(f'**/[a-z|0-9]*.{filetype}')
    # In my labels file weren't the file suffixes, please change to your needs
    return {t.stem : t for t in train_files}

def getFnamesAndLabels(csv_file,skip_header=True):
    import numpy as np
    # I assume in the labels file, that the first column are the filenames and all following columns are the labels
    # Therefore, more flexibility when using multicategory (more than 2 classes)
    labels = np.genfromtxt(csv_file, skip_header=int(skip_header), converters={0: lambda x: np.str(x.decode('utf-8'))}, delimiter=',')
    return labels

def createNewMemmapFiles(mode,sequence_dtype,labels_dtype):
    import numpy as np
    data = np.memmap(f'{mode}.dat',sequence_dtype,'w+',shape=(1))
    labels = np.memmap(f'{mode}_meta.dat',labels_dtype,'w+',shape=(1))
    del data
    del labels
    
# The convert process: reads the old data and safes it a new files
def safeOldDataToMemMap(files_dict,labels,mode,data_folder,sequence_dtype,labels_dtype):
    import numpy as np
    from tqdm import tqdm
    
    old_data_length = old_meta_length = 0
    data_bytes = np.dtype(sequence_dtype).itemsize
    labels_bytes = np.dtype(labels_dtype).itemsize
    
    def writeData(data,meta_data,old_data_length,old_meta_length):
        data_mapped = np.memmap(f'{mode}.dat',sequence_dtype,'r+',shape=(data_to_write),offset=data_bytes*old_data_length)
        data_mapped[:] = data
        del data_mapped
        
        meta_mapped = np.memmap(f'{mode}_meta.dat',labels_dtype,'r+',shape=(meta_data_to_write),offset=labels_bytes*old_meta_length)
        meta_mapped[:] = meta_data
        del meta_mapped
    
    def readOldData(filename):
        return np.loadtxt(filename,dtype=sequence_dtype,delimiter=',')
        
    for label in tqdm(labels):
        name, *counts = label
        try:
            files_dict[name]
        except KeyError:
            print(f'File (stem) {name} does not exist')
            continue
        
        data = readOldData(files_dict[name])
        sequence_length = data.shape[0]
        meta_data = np.asarray([sequence_length,*counts],dtype=labels_dtype)
        
        data = data.flatten()
        meta_data = meta_data.flatten()
       
        data_to_write = data.shape[0]    
        meta_data_to_write = meta_data.shape[0]
        
        writeData(data,meta_data,old_data_length,old_meta_length)

        old_data_length+=data_to_write
        old_meta_length+=meta_data_to_write

if __name__=='__main__':
    # just do it for all the files/modes (training/validation/testing)
    mode = 'training'
                         
    from utils import loadConfig
    data_parameter, *_ = loadConfig(verbose=0)
                         
    data_folder = data_parameter["data directory"]
    sequence_dtype = data_parameter["sequence dtype"]
    labels_dtype = data_parameter["labels dtype"]
    
    legacy_labels = f'data/labels/{data_parameter["%s label"%mode]}'
    legacy_data_directory = 'data/csv/'
    legacy_file_type = 'csv'
    
    files_dict = getNamesOfAllDataFiles(legacy_data_directory,filetype=legacy_file_type)
    labels = getFnamesAndLabels(legacy_labels)
    createNewMemmapFiles(mode=mode,sequence_dtype=sequence_dtype,labels_dtype=labels_dtype)
    # As a sidenote: The data is saved as the file names appear in the 'labels' array
    safeOldDataToMemMap(files_dict,labels,mode,data_folder,sequence_dtype,labels_dtype)