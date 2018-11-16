from time import time
from embedding.wordEmbeddings import createEmbeddingMatrixFromModel, convertGloveToWord2VecModel, loadWord2VecModel
from preprocess.fileutil import loadCSV, saveDataFrameToCSV
from preprocess.tokenutil import cleanColumn, tokenizeComments
from preprocess.preprocess import getVocabSize

EMBEDDING_DIMENSIONS = 25
MAX_WORDS = 200000

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
  # Load training data
  trainData = loadCSV(cleanedTrainFile, ['id', 'comment_text'])
  trainComments = trainData['comment_text']

  # Load pre-trained word embeddings
  convertGloveToWord2VecModel(gloveFile, word2vecFile)
  model = loadWord2VecModel(word2vecFile)
  # Create embedding matrix from the pre-trained word embeddings and train data
  trainMatrix = createEmbeddingMatrixFromModel(trainComments, model, MAX_WORDS, EMBEDDING_DIMENSIONS)

  # trainLabels = loadCSV(trainFile, ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
  
  print('Done! Completed in', time() - startTime, 'seconds')


if __name__ == "__main__":
  main()