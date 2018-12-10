import argparse
from time import time
from predict import predictTestData
from train import train
from server import start, loadModel
from model.cnn import KerasCNN
from model.lstm import KerasLSTM
from preprocess.fileutil import loadCSV

def main():
  startTime = time()
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train', action='store_true')
  parser.add_argument('-p', '--predict', action='store_true')
  parser.add_argument('-s', '--serve', action='store_true')
  parser.add_argument('-bs', '--batchsize', type=int)
  parser.add_argument('-ep', '--epoch', type=int)
  parser.add_argument('-ed', '--embeddim', type=int)
  args = parser.parse_args()
  if(not args.train and not args.predict and not args.serve):
    print('--train, --predict, and/or --serve must be specified')
    return

  if(not args.batchsize or not args.epoch or not args.embeddim):
    print('--batchsize, --epoch, and --embeddim must be specified')
    return
  cnnModelFile = 'model/cnnModel.h5'
  lstmModelFile =  'model/lstmModel.h5'
  modelFile = cnnModelFile
  cleanedTrainFile = 'data/train_cleaned.csv'
  model = KerasCNN()
  # [-t | --train]
  if(args.train):
    trainFile = 'data/train.csv'
    gloveFile = 'data/glove.twitter.27B.'+str(args.embeddim)+'d.txt'
    word2vecFile = 'data/word2vec.twitter.27B.'+str(args.embeddim)+'d.txt'
    train(model, trainFile, cleanedTrainFile, gloveFile, word2vecFile, modelFile, args)
  
  # [-p | --predict] -- predict excel file
  if(args.predict):
    model.loadModel(modelFile)
    print('Loading trained model from', modelFile, ' ...')
    testFile = 'data/test.csv'
    cleanedTestFile = 'data/test_cleaned.csv'
    predictTestData(testFile, cleanedTestFile, model)

  # [-s | --serve] -- serve/host the model for real-time predictions
  # Tokenizer must already be created. If not, --train must be called.
  if(args.serve):
    loadModel(model, modelFile)
    start()
  print('Done! Completed in', time() - startTime, 'seconds')

if __name__ == "__main__":
  main()