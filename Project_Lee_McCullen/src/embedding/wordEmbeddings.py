from smart_open import smart_open
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
import numpy as np

"""
  Convert the .txt file containing Glove's pre-trained embeddings
  into a .txt file that is readable by Word2Vec.
"""
def convertGloveToWord2VecModel(gloveFileName, outputFileName):
  print("Converting", gloveFileName, "Embeddings to Word2Vec...")
  try:
    glove2word2vec(gloveFileName, outputFileName)
  except OSError:
    print(gloveFileName, 'does not exist. Please download from https://nlp.stanford.edu/projects/glove/')
  print('Word2Vec model saved in', outputFileName)

"""
  Load Word2Vec model.
"""
def loadWord2VecModel(fileName):
  model = KeyedVectors.load_word2vec_format(fileName)
  return model

"""
  Create a unique index (dictionary) of all words
"""
def createWordIndex(data, maxWords):
  tokenizer = Tokenizer(num_words=maxWords)
  tokenizer.fit_on_texts(data)
  return tokenizer.word_index

"""
  Create embedding matrix from pre-trained word embedding model.
  Currently using pre-trained Glove embeddings of Tweets
"""
def createEmbeddingMatrixFromModel(data, model, maxWords, embeddingDimensions):
  wordIndex = createWordIndex(data, maxWords)
  # Keep the top N words in our vocab
  vocabSize = min(len(wordIndex)+1, maxWords)
  # Initialize embedding matrix
  embeddingMatrixShape = (vocabSize, embeddingDimensions)
  embeddingMatrix = np.zeros(embeddingMatrixShape)

  # Create embedding matrix
  for word, idx in wordIndex.items():
    if idx >= maxWords:
      continue
    try:
      vector = model[word]
      embeddingMatrix[idx] = vector
    except KeyError:
      # Handle unknown word vectors by giving it a random value
      vector = np.random.rand(embeddingDimensions)
      embeddingMatrix[idx] = vector

  print(embeddingMatrix.shape)
  return embeddingMatrix

"""
  Creates word embeddings (Word2Vec) from scratch given a corpus.
"""
def createEmbeddingsFromCorpus(fileName):
  #Preprocessing data
  trainData = pd.read_csv(fileName)
  comments = trainData["comment_text"]

  tokenizedRows = []

  for c in comments:
      row = word_tokenize(c)
      tokenizedRows.append(row)

  #Create model from preprocessed data
  model = Word2Vec(tokenizedRows, workers=4)

  #Summarize vocab
  words = list(model.wv.vocab)

  return model
  #Save model for later use
  #model.save('wordModel.bin')

