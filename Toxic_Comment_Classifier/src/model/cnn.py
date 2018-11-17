from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input
from keras.layers.embeddings import Embedding

class CNN:
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

  def createModel(self, embeddingMatrix, vocabSize, embeddingDimensions, tokenLength):
    print('Instantiating Keras CNN model...')

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

    # Regularization layer to prevent overfitting
    dropoutLayer = Dropout(rate=0.5)(poolingLayer)

    ### OUTPUT LAYER ###

    # Flatten output into 1D feature vector
    flattenLayer = Flatten()(dropoutLayer)
    outputLayer = Dense(6, activation='sigmoid')(flattenLayer)

    ### CREATE MODEL ###

    self.model = Model(mainInputLayer, outputLayer)
    self.model.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['acc'])
    self.model.summary()

    print('Keras model successfully instantiated...')

  def fit(self, X, Y, epochs, batchSize):
    if(self.model is None):
      raise Exception("Model is undefined. Call createModel() first before fitting.")
    print('Fitting model...')
    self.model.fit(
      x=X, 
      y=Y, 
      epochs=epochs, 
      validation_split=0.2, 
      shuffle=True, 
      batch_size=batchSize)

  def saveModel(self, outputFile):
    if(self.model is None):
      raise Exception("Undefined models cannot be saved.")
    print('Saving model...')
    self.model.save(outputFile)
    print('Model successfully saved to', outputFile)

  def loadModel(self, inputFile):
    print('Loading model...')
    self.model = load_model(inputFile)
    print(inputFile, 'model loaded...')