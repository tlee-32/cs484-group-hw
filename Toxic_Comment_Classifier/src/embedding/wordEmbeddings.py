from smart_open import smart_open
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import bcolz
import pandas as pd
import pickle
import numpy as np

"""
  Load and save pre-trained embeddings from Glove.
"""
def processPretrainedEmbeddings(fileName):
  print("Loading", fileName, "Embeddings")
  idx = 0
  words = []
  wordToIdx = {}
  embeddings = bcolz.carray(np.zeros(1), rootdir=fileName+'.dat', mode='w')

  # Load embeddings
  with smart_open(fileName+'.txt', 'rb') as f:
      for l in f:
          line = l.decode().split()
          word = line[0]
          words.append(word)
          wordToIdx[word] = idx
          idx += 1
          vector = np.array(line[1:]).astype(np.float)
          embeddings.append(vector)
      
  # Save embeddings to disk
  shape = (-1, 50) # 400K tokens and 50 dimensions
  embeddings = bcolz.carray(embeddings[1:].reshape(shape), rootdir=fileName+'.dat', mode='w')
  embeddings.flush()
  # Save pre-trained corpus
  pickle.dump(words, smart_open(fileName+'_words.pkl', 'wb'))
  pickle.dump(wordToIdx, smart_open(fileName+'_idx.pkl', 'wb'))
  print('Completed processing ', fileName+'.txt')

"""
  Loads the embeddings from the files.
"""
def loadSavedEmbeddings(fileName):
  embeddings = bcolz.open(fileName+'.dat')[:]
  words = pickle.load(smart_open(fileName+'_words.pkl', 'rb'))
  wordToIdx = pickle.load(smart_open(fileName+'_idx.pkl', 'rb'))

  glove = {word: embeddings[wordToIdx[word]] for word in words}
  return glove

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
  print(words)
  print(model)

  #Save model for later use
  #model.save('wordModel.bin')

