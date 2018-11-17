from time import time
from embedding.wordEmbeddings import createEmbeddingMatrixFromModel, convertGloveToWord2VecModel, loadWord2VecModel
from preprocess.fileutil import loadCSV, saveDataFrameToCSV
from preprocess.tokenutil import cleanColumn, tokenizeComments, createPaddedTokens
from preprocess.preprocess import getVocabSize
from model.cnn import KerasCNN

EMBEDDING_DIMENSIONS = 25
MAX_TOKEN_LENGTH = 250
MAX_WORDS = 200000
FILTER_WINDOWS = [3,4,5]
FEATURE_MAP_SIZE = 100

def main():
  startTime = time()
  trainFile = 'data/train.csv'
  cleanedTrainFile = 'data/train_cleaned.csv'
  testFile = 'data/test.csv'
  gloveFile = 'data/glove.twitter.27B.25d.txt'
  word2vecFile = 'data/word2vec.twitter.27B.25d.txt'
  """
  # Load, clean, and save the training data
  trainData = loadCSV(trainFile, ['id', 'comment_text'])
  trainData['comment_text'] = cleanColumn(trainData['comment_text'])
  saveDataFrameToCSV(cleanedTrainFile, trainData)
  """

  print('Loading train data...')
  # Load training data
  trainData = loadCSV(cleanedTrainFile, ['id', 'comment_text'])
  trainComments = trainData['comment_text'].tolist()
  trainLabels = loadCSV(trainFile, ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
  
  print('Loading validation data...')


  # Load pre-trained word embeddings
  print('Loading pre-trained embeddings...')
  #convertGloveToWord2VecModel(gloveFile, word2vecFile)
  model = loadWord2VecModel(word2vecFile)
  
  # Create embedding matrix from the pre-trained word embeddings and train data
  print('Creating embedding matrix...')
  embeddingMatrix = createEmbeddingMatrixFromModel(trainComments, model, MAX_WORDS, EMBEDDING_DIMENSIONS)
  tokenizedTrainComments = createPaddedTokens(trainComments, MAX_WORDS, MAX_TOKEN_LENGTH)

  print('Instantiating Keras CNN model...')

  print('Keras model successfully instantiated...')

  print('Fitting model...')
  
  # cnn = new KerasCNN(FILTER_WINDOWS, FEATURE_MAP_SIZE)
  # cnn.fit(tokenizedTrainComments, trainLabels.values, ...)
  # cnn.saveModel('model/cnnModel.h5')
  # cnn.loadModel('model/cnnModel.h5')
  
  # TODO: Create tokenized test comments and predict them
  # cnn.predict()...
  
  print('Done! Completed in', time() - startTime, 'seconds')


if __name__ == "__main__":
  main()