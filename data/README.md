This folder contains the dataset to train/validate/test the model.

Some explanation to the files:

* valid.csv
    * kept just as reference for the data conversion, how the labels file should look like 

* the data is saved in a binary format for efficient loading (using numpy memory-mapping)
* no header or structure information included (partial loading of the files is possible)

* validation.dat
    * contains all validation sequences which are provided to the public
* validation_meta.dat
    * contains the sequence labels and lengths (in tuples of 3)
    * first entry is the sequence length (to load from the validation.dat)
    * next 2 entries are the labels (only for the last fram of each sequence)
* The other files follow the same naming scheme