import numpy as np
import csv
import shutil

with open('labels_public/valid.csv','r') as f:
  filenames = np.asarray(list(csv.reader(f)))[:,0]
  for filename in filenames:
    name = filename+'.csv'
    shutil.move('./csv/%s'%name,'./csv_public/%s'%name)
