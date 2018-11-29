import argparse
from time import time
from predict import predictTestData
from train import train
from server import start, loadModel
from model.cnn import KerasCNN
from preprocess.fileutil import loadCSV

MAX_TOKEN_LENGTH = 100
FILTER_WINDOWS = [3,4,5]
FEATURE_MAP_SIZE = 100

def main():
  startTime = time()
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train', action='store_true')
  parser.add_argument('-p', '--predict', action='store_true')
  parser.add_argument('-s', '--serve', action='store_true')
  args = parser.parse_args()
  if(not args.train and not args.predict and not args.serve):
    print('--train, --predict, and/or --serve must be specified')
    return

  modelFile = 'model/cnnModel.h5'
  # modelFile = 'model/weights_base.best.hdf5'
  cleanedTrainFile = 'data/train_cleaned.csv'
  # [-t | --train]
  if(args.train):
    trainFile = 'data/train.csv'
    gloveFile = 'data/glove.twitter.27B.25d.txt'
    word2vecFile = 'data/word2vec.twitter.27B.25d.txt'
    train(trainFile, cleanedTrainFile, gloveFile, word2vecFile, modelFile)

  cnn = KerasCNN()
  cnn.loadModel(modelFile)
  # [-p | --predict] -- predict excel file
  if(args.predict):
    print('Loading trained model from', modelFile, ' ...')
    testFile = 'data/test.csv'
    cleanedTestFile = 'data/test_cleaned.csv'
    predictTestData(testFile, cleanedTestFile, cnn)

  # [-s | --serve] -- serve/host the model for real-time predictions
  # Tokenizer must already be created. If not, --train must be called.
  if(args.serve):
    loadModel(cnn, modelFile)
    start()
  print('Done! Completed in', time() - startTime, 'seconds')

if __name__ == "__main__":
  main()