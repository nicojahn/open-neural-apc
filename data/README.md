This folder contains all labels and data used to train/test/validate the model.
Otherwise it is not very interesting.

Some explanation to folders/files:

csv
  not intended for the public (not my decision)
csv_public
  intended for the public (for testing and imagination how the data looks like)
labels
  not intended for the public (not my decision)
labels_public
  intended for the public (for testing and imagination how the data looks like)
rename.py
  renamed the original files into theire hashes ('anonymization')
split.py
  chooses randomly 15 files and creates new label files (train/valid)
move_public_csv.py
  moves the files which are dedicated for public access into correspondig folder
