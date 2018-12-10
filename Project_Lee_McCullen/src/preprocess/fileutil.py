import pandas as pd
import os.path
import pickle

"""
  Read structured file into a DataFrame.
"""
def loadCSV(file, columns):
  df = pd.read_csv(file, sep=',', low_memory=False, header=0)
  return df[columns]

def saveDataFrameToCSV(file, df):
  df.to_csv(file, index=False)

def saveTokenizer(tokenizer, file):
  with open(file, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadTokenizer(file):
  with open(file, 'rb') as handle:
    return pickle.load(handle)

def fileExists(file):
  return os.path.exists(file)