from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from .metrics import auc_roc

"""
  Convolutional Neural Network using Keras
"""
class KerasCNN:
  def __init__(self, filterWindows, featureMapSize):
    self.model = None
    # List holding the length of 1D convolution windows
    self.filterWindows = filterWindows
    # Number of output filters for a convolution (i.e. feature map)
    self.featureMapSize = featureMapSize
    self.model = None
  
  def createEmbeddingLayer(self, embeddingMatrix, vocabSize, embeddingDimensions, tokenLength):
    embeddingLayer = Embedding(
      input_dim = vocabSize, 
      output_dim = embeddingDimensions,
      weights = [embeddingMatrix],
      input_length = tokenLength)
    return embeddingLayer

  """
    Build the CNN model with the appropriate parameters
  """
  def createModel(self, embeddingMatrix, vocabSize, embeddingDimensions, tokenLength):
    ### INPUT LAYER ###

    # Embedding layer will expect batches of tokenLength-dimensional vectors
    mainInputLayer = Input(shape=(tokenLength,), name='mainInput')
    
    # Encode the input sequence into a sequence of dense tokenLength-dimensional vectors.
    embeddingLayer = self.createEmbeddingLayer(embeddingMatrix, vocabSize, embeddingDimensions, tokenLength)
    embeddingLayer = embeddingLayer(mainInputLayer)
    
    ### HIDDEN LAYER 1 ###

    # Convolution layer with a 3-window convolution/filter
    convolutionLayer = Conv1D(
      filters=self.featureMapSize,
      kernel_size=3,
      activation='relu')(embeddingLayer)
    
    # Max pooling
    poolingLayer = MaxPooling1D(pool_size=3)(convolutionLayer)
  
    ### FULLY CONNECTED LAYER ###

    # Flatten output into 1D feature vector
    flattenLayer = Flatten()(poolingLayer)
    
    # Regularization layer to prevent overfitting
    dropoutLayer = Dropout(rate=0.5)(flattenLayer)

    ### OUTPUT LAYER ###

    outputLayer = Dense(6, activation='sigmoid')(dropoutLayer)

    ### CREATE MODEL ###

    self.model = Model(mainInputLayer, outputLayer)
    self.model.compile(
      loss='categorical_crossentropy',
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
    earlyStopping = EarlyStopping(monitor='auc_roc', patience=2)
    # Train model
    self.model.fit(
      x=X, 
      y=Y, 
      epochs=epochs, 
      validation_split=0.2, 
      shuffle=True, 
      batch_size=batchSize,
      callbacks=[earlyStopping])

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
    # Keras doesn't save custom loss functions so we have to define it
    self.model = load_model(inputFile, custom_objects={'auc_roc': auc_roc})
    print(inputFile, 'model loaded...')