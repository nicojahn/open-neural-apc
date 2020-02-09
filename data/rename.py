import numpy as np
import csv
import shutil
import hashlib

with open('./labels/labels.csv','r') as f:
  labels = np.asarray(list(csv.reader(f)))
  filenames = np.asarray(labels[:,:1],dtype=str)
  label = labels[:,1:]
  fn = []
  
  for idx,name in enumerate(filenames[:,0]):
    hash_name = (hashlib.sha1(name).hexdigest()) 
    shutil.copy('./csv/%s.csv'%name,'./csv/%s.csv'%(hash_name))
    fn += [str(hash_name)]

  matrix = np.concatenate([np.asarray(fn)[:,np.newaxis],label],axis=1)

  with open('./labels/labels.csv','w+') as file:
    writer = csv.writer(file)
    writer.writerows(matrix)
