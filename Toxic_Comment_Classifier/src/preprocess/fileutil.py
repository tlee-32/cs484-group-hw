import pandas as pd

"""
  Read structured file into a DataFrame.
"""
def loadCSV(file, columns):
  df = pd.read_csv(file, sep=',', low_memory=False, header=0)
  return df[columns]

def saveDataFrameToCSV(file, df):
  df.to_csv(file)