import pickle
import pandas as pd

"""
  Read structured file into a DataFrame.
"""
def readFile(file, separator, columns, types):
  return pd.read_csv(file, sep=separator, names=columns, low_memory=False, dtype=types, header=0)