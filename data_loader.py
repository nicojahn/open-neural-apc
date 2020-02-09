# Copyright (c) 2020, Nico Jahn
# All rights reserved.

import numpy as np
import os
import csv

class loader():
    def __init__(self, parent_directory,frame_hops=1):
        self.big_list = []
        self.parent_directory = parent_directory
        self.frame_hops = frame_hops
    
    def load_file_from_list(self, filenames):
        X = []
        y = np.asarray(np.asarray(filenames)[:,1:3],dtype=int)

        for counter,entry in enumerate(filenames):
            filename = entry[0]
            # read the sequence from the csv file
            with open('%s%s.csv'%(self.parent_directory,filename), newline='') as f:
                reader = list(csv.reader(f, delimiter=','))[::self.frame_hops]
                # rescale into range [0,1]
                X += [np.asarray(reader, dtype=float)/255.]
                
            if counter % 50 == 0:
                print(counter)
        return X, y
