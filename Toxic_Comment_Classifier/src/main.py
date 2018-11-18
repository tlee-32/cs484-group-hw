import argparse
import os
from time import time
from embedding.wordEmbeddings import createEmbeddingMatrixFromModel, convertGloveToWord2VecModel, loadWord2VecModel
from preprocess.fileutil import loadCSV, saveDataFrameToCSV, fileExists
from preprocess.tokenutil import cleanColumn, tokenizeComments, createPaddedTokens
from preprocess.preprocess import getVocabSize
from model.cnn import KerasCNN

EMBEDDING_DIMENSIONS = 25
MAX_TOKEN_LENGTH = 250
MAX_VOCAB_SIZE = 200000
FILTER_WINDOWS = [3,4,5]
FEATURE_MAP_SIZE = 100

"""
  Trains the CNN model and saves it to a file.
"""
def train(trainFile, cleanedTrainFile, gloveFile, word2vecFile, modelOutputFile):
  print('Loading train data...')
  trainData = ''
  # Load, clean, and save the training data
  if(not fileExists(cleanedTrainFile)):
    trainData = loadCSV(trainFile, ['id', 'comment_text'])
    trainData['comment_text'] = cleanColumn(trainData['comment_text'])
    saveDataFrameToCSV(cleanedTrainFile, trainData)
  else:
    trainData = loadCSV(cleanedTrainFile, ['id', 'comment_text'])
  
  trainComments = trainData['comment_text'].tolist()
  trainLabels = loadCSV(trainFile, ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

  # Load pre-trained word embeddings
  print('Loading pre-trained embeddings...')
  if(not fileExists(word2vecFile)):
    print(word2vecFile, 'model was not found. Performing conversion...')
    convertGloveToWord2VecModel(gloveFile, word2vecFile)
  model = loadWord2VecModel(word2vecFile)
  
  # Create embedding matrix from the pre-trained word embeddings and train data
  print('Creating embedding matrix...')
  embeddingMatrix = createEmbeddingMatrixFromModel(trainComments, model, MAX_VOCAB_SIZE, EMBEDDING_DIMENSIONS)
  vocabSize = embeddingMatrix.shape[0]
  tokenizedTrainComments = createPaddedTokens(trainComments, vocabSize, MAX_TOKEN_LENGTH)

  print('Instantiating Keras CNN model...')
  cnn = KerasCNN(FILTER_WINDOWS, FEATURE_MAP_SIZE)
  cnn.createModel(embeddingMatrix, vocabSize, EMBEDDING_DIMENSIONS, MAX_TOKEN_LENGTH)

  print('Fitting model...')
  cnn.fit(X=tokenizedTrainComments, Y=trainLabels.values, epochs=2, batchSize=50)

  cnn.saveModel(modelOutputFile)

def predict(testFile, modelInputFile):
  print('Loading trained model from', modelInputFile, ' ...')
  cnn = KerasCNN(FILTER_WINDOWS, FEATURE_MAP_SIZE)
  cnn.loadModel(modelInputFile)
  # TODO: Create tokenized test comments and predict them
  # cnn.predict()...
  return 0

def main():
  startTime = time()
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train', action='store_true')
  parser.add_argument('-p', '--predict', action='store_true')
  args = parser.parse_args()
  if(not args.train and not args.predict):
    print('--train and/or --predict must be specified')
    return

  modelFile = 'model/cnnModel.h5'
  # [-t | --train]
  if(args.train):
    trainFile = 'data/train.csv'
    cleanedTrainFile = 'data/train_cleaned.csv'
    gloveFile = 'data/glove.twitter.27B.25d.txt'
    word2vecFile = 'data/word2vec.twitter.27B.25d.txt'
    train(trainFile, cleanedTrainFile, gloveFile, word2vecFile, modelFile)

  # [-p | --predict]
  if(args.predict):
    testFile = 'data/test.csv'
    predict(testFile, modelFile)

  print('Done! Completed in', time() - startTime, 'seconds')


def writePredictions(predictions):
  return 0
if __name__ == "__main__":
  main()