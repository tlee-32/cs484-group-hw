import pandas as pd
import numpy as np
from preprocess.fileutil import loadCSV, saveDataFrameToCSV, fileExists
from preprocess.tokenutil import cleanColumn, createPaddedTokens
from model.cnn import KerasCNN

MAX_TOKEN_LENGTH = 100

def predictTestData(testFile, cleanedTestFile, model):
  print('Loading test data...')
  df = ''
  # Load, clean, and save the training data
  if(not fileExists(cleanedTestFile)):
    df = loadCSV(testFile, ['id', 'comment_text'])
    df['comment_text'] = cleanColumn(df['comment_text'])
    saveDataFrameToCSV(cleanedTestFile, df)
  else:
    df = loadCSV(cleanedTestFile, ['id', 'comment_text'])

  testComments = df['comment_text'].tolist()
  tokenizedTestComments = createPaddedTokens(testComments, MAX_TOKEN_LENGTH, isTrainingFile=False)
  
  predictions = model.predict(tokenizedTestComments)
  classNames = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

  savePredictions(predictions, df, classNames)

def savePredictions(predictions, df, classNames):
  columns = ['id'] + classNames
  predictionDF = pd.DataFrame(columns=columns)
  predictionDF['id'] = df['id'].tolist() 
  predictionDF[classNames] = predictions
  saveDataFrameToCSV('./data/predictions.csv', predictionDF)
