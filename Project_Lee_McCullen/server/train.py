from embedding.wordEmbeddings import createEmbeddingMatrixFromModel, convertGloveToWord2VecModel, loadWord2VecModel
from preprocess.fileutil import loadCSV, saveDataFrameToCSV, fileExists
from preprocess.tokenutil import cleanColumn, createPaddedTokens, createAndSaveTokenizer
from model.cnn import KerasCNN

EMBEDDING_DIMENSIONS = 25
MAX_TOKEN_LENGTH = 100
MAX_VOCAB_SIZE = 193491
EPOCHS = 4
BATCHSIZE = 32
"""
  Trains the CNN model and saves it to a file.
"""
def train(trainFile, cleanedTrainFile, gloveFile, word2vecFile, modelOutputFile):
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
  model = loadWord2VecModel(word2vecFile)
  
  # Create embedding matrix from the pre-trained word embeddings and train data
  print('Creating embedding matrix...')
  embeddingMatrix = createEmbeddingMatrixFromModel(trainComments, model, MAX_VOCAB_SIZE, EMBEDDING_DIMENSIONS)
  
  print('Fitting and saving training data on tokenizer...')
  createAndSaveTokenizer(trainComments, MAX_VOCAB_SIZE)
  tokenizedTrainComments = createPaddedTokens(trainComments, MAX_TOKEN_LENGTH, isTrainingFile=True)
  
  # Create model
  print('Instantiating Keras CNN model...')
  cnn = KerasCNN()
  cnn.createModel(embeddingMatrix, MAX_VOCAB_SIZE, EMBEDDING_DIMENSIONS, MAX_TOKEN_LENGTH)
  
  print('Fitting model...')
  cnn.fit(X=tokenizedTrainComments, Y=trainLabels.values, epochs=EPOCHS, batchSize=BATCHSIZE)

  cnn.saveModel(modelOutputFile)