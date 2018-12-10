import pandas as pd
import numpy as np

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

def toxicCount():
  classNames = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
  toxicCounts = []

  for className in classNames:
    count = train.groupby(className)['id'].count()[1]
    toxicCounts.append(count)

  return toxicCounts

def nonToxicCount():
  classNames = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
  return [train.groupby(classNames)['id'].count().values[0]]
