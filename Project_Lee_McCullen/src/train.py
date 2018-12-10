from embedding.wordEmbeddings import createEmbeddingMatrixFromModel, convertGloveToWord2VecModel, loadWord2VecModel
from preprocess.fileutil import loadCSV, saveDataFrameToCSV, fileExists
from preprocess.tokenutil import cleanColumn, createPaddedTokens, createAndSaveTokenizer

EMBEDDING_DIMENSIONS = 100
MAX_TOKEN_LENGTH = 100
MAX_VOCAB_SIZE = 193491

"""
  Trains the CNN model and saves it to a file.
"""
def train(model, trainFile, cleanedTrainFile, gloveFile, word2vecFile, modelOutputFile, args):
  EPOCHS = args.epoch
  BATCHSIZE = args.batchsize
  EMBEDDING_DIMENSIONS = args.embeddim
  print('Loading train data...')
  df = ''
  # Load, clean, and save the training data
  if(not fileExists(cleanedTrainFile)):
    df = loadCSV(trainFile, ['id', 'comment_text'])
    df['comment_text'] = cleanColumn(df['comment_text'])
    saveDataFrameToCSV(cleanedTrainFile, df)
  else:
    df = loadCSV(cleanedTrainFile, ['id', 'comment_text'])
  
  trainComments = df['comment_text'].tolist()
  trainLabels = loadCSV(trainFile, ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

  # Load pre-trained word embeddings
  print('Loading pre-trained embeddings...')
  if(not fileExists(word2vecFile)):
    print(word2vecFile, 'model was not found. Performing conversion...')
    convertGloveToWord2VecModel(gloveFile, word2vecFile)
  word2VecModel = loadWord2VecModel(word2vecFile)
  
  # Create embedding matrix from the pre-trained word embeddings and train data
  print('Creating embedding matrix...')
  embeddingMatrix = createEmbeddingMatrixFromModel(trainComments, word2VecModel, MAX_VOCAB_SIZE, EMBEDDING_DIMENSIONS)
  
  print('Fitting and saving training data on tokenizer...')
  createAndSaveTokenizer(trainComments, MAX_VOCAB_SIZE)
  tokenizedTrainComments = createPaddedTokens(trainComments, MAX_TOKEN_LENGTH, isTrainingFile=True)
  
  # Create model
  print('Instantiating Keras model...')
  model.createModel(embeddingMatrix, MAX_VOCAB_SIZE, EMBEDDING_DIMENSIONS, MAX_TOKEN_LENGTH)
  print('Fitting model...')
  model.fit(X=tokenizedTrainComments, Y=trainLabels.values, epochs=EPOCHS, batchSize=BATCHSIZE)

  model.saveModel(modelOutputFile)