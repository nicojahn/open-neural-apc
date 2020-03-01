import numpy as np
import csv

with open('./labels/labels.csv','r') as f:
  labels = np.asarray(list(csv.reader(f)))
  indices = np.arange(labels.shape[0])
  selection = np.random.choice(indices, 15)
  print(labels[selection])

  with open('./labels_public/valid.csv','w+') as file:
    writer = csv.writer(file)
    writer.writerows(labels[selection])

  with open('./labels/train.csv','w+') as file:
    writer = csv.writer(file)
    indices = np.delete(indices,selection)
    writer.writerows(labels[indices])
