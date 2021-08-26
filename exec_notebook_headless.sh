#!/bin/bash
jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 --inplace open-neural-apc.ipynb
