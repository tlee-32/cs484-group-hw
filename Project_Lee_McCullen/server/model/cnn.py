from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate, Bidirectional, GRU, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from .metrics import auc_roc

FILTER_WINDOWS = [3,4,5] # List holding the length of 1D convolution windows
FEATURE_MAP_SIZE = 100 # Number of output filters for a convolution (i.e. feature map)

"""
  Convolutional Neural Network using Keras
"""
class KerasCNN:
  def __init__(self):
    self.model = None
  
  """
    Create embedding layer from scratch
  """
  def createEmbeddingLayer(self, vocabSize, embeddingDimensions, tokenLength):
    embeddingLayer = Embedding(
      input_dim = vocabSize, 
      output_dim = embeddingDimensions,
      input_length = tokenLength)
    return embeddingLayer

  """
    Create embedding layer from pre-trained vectors
  """
  def createPreTrainedEmbeddingLayer(self, embeddingMatrix, vocabSize, embeddingDimensions, tokenLength):
    embeddingLayer = Embedding(
      input_dim = vocabSize, 
      output_dim = embeddingDimensions,
      weights = [embeddingMatrix],
      input_length = tokenLength,
      trainable=False)
    return embeddingLayer

  """
    Build the CNN model with the appropriate parameters
  """
  def createModel(self, embeddingMatrix, vocabSize, embeddingDimensions, tokenLength):
    ### INPUT LAYER ###

    # Embedding layer will expect batches of tokenLength-dimensional vectors
    mainInputLayer = Input(shape=(tokenLength,), name='mainInput')
    
    embeddingLayer = self.createPreTrainedEmbeddingLayer(embeddingMatrix, vocabSize, embeddingDimensions, tokenLength)
    # Encode the input sequence into a sequence of dense tokenLength-dimensional vectors.
    embeddingLayer = embeddingLayer(mainInputLayer)
    
    ### HIDDEN LAYER ###

    # convolutions = []

    # for filterWindow in FILTER_WINDOWS:
    #     convolutionLayer = Conv1D(filters=128, kernel_size=filterWindow, activation='relu')(embeddingLayer)
    #     poolingLayer = MaxPooling1D(pool_size=3)(convolutionLayer)
    #     convolutions.append(poolingLayer)

    # mergedConvolutions = concatenate(convolutions, axis=1)

    # Convolution layer with 300-filters of size 4 each
    convolutionLayer = Conv1D(
      filters=300, # 
      padding='same', # pads so that input is same size as output
      kernel_size=4, # size of sliding-window (context window of 4 words OR word embeddings of 4 words)
      activation='relu')(embeddingLayer)
    
    # Max pooling
    poolingLayer = MaxPooling1D(pool_size=4)(convolutionLayer)

    # Regularization layer to prevent overfitting part 1
    dropoutLayerOne = Dropout(rate=0.2)(poolingLayer)

    ### FULLY CONNECTED LAYER ###
    # Flatten output into 1D feature vector
    flattenLayer = Flatten()(dropoutLayerOne)
    
    denseLayer = Dense(300, activation='relu')(flattenLayer)

    # Regularization layer to prevent overfitting part 2
    dropoutLayerTwo = Dropout(rate=0.2)(denseLayer)

    ### OUTPUT LAYER ###

    outputLayer = Dense(6, activation='sigmoid')(dropoutLayerTwo)

    ### CREATE MODEL ###

    self.model = Model(mainInputLayer, outputLayer)
    self.model.compile(
      loss='binary_crossentropy',
      optimizer='adam',
      metrics=['acc', auc_roc])
    self.model.summary()

  """
    Train the CNN
  """
  def fit(self, X, Y, epochs, batchSize):
    if(self.model is None):
      raise Exception("Model is undefined. Call createModel() first before fitting.")
    # Interrupt training when validation loss stops decreasing.
    earlyStopping = EarlyStopping(monitor='auc_roc', mode='min', patience=15)
    # Train model
    self.model.fit(
      x=X, 
      y=Y, 
      epochs=epochs, 
      validation_split=0.1, 
      batch_size=batchSize,
      callbacks=[earlyStopping])

  def predict(self, testData, batchSize=32):
    print('Predicting...')
    return self.model.predict(testData, batch_size=batchSize, verbose=1)

  """
    Save the CNN model to a file
  """
  def saveModel(self, outputFile):
    if(self.model is None):
      raise Exception("Undefined models cannot be saved.")
    print('Saving model...')
    self.model.save(outputFile)
    print('Model successfully saved to', outputFile)

  """
    Load the CNN model from a file
  """
  def loadModel(self, inputFile):
    print('Loading model...')
    # Keras doesn't save custom loss functions so we have to load it back.
    self.model = load_model(inputFile, custom_objects={'auc_roc': auc_roc})
    # self.model.load_weights(inputFile)
    print(inputFile, 'model loaded...')