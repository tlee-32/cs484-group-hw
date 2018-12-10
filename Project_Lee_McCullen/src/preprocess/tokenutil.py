import re
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from .fileutil import saveTokenizer, fileExists, loadTokenizer

def cleanColumn(dfCol):
  dfCol = dfCol.replace(to_replace=r'http', value='', regex=True)
  dfCol = dfCol.replace(to_replace=r'@', value='at', regex=True) 
  dfCol = dfCol.replace(to_replace=r'[!#$%&()*+,-.\/:;<=>?[\]^_`{|}~\'\`\"\_\n\\]', value=' ', regex=True)  
  dfCol = dfCol.str.lower()
  return dfCol

def cleanText(text):
  text = text.replace('http', '')
  text = text.replace('@', 'at')
  text = re.sub(r'[!#$%&()*+,-.\/:;<=>?[\]^_`{|}~\'\`\"\_\n\\]', ' ', text)
  return text
"""
  Apply tokenize function on comments.
"""
def tokenizeComments(commentColumn):
  result = []
  comments = commentColumn.tolist()
  for comment in comments:
    result.append(word_tokenize(comment))
  return result

def tokenizeText(text):
  return word_tokenize(text)
"""
  Fit training data on tokenizer and pickle it
  train - 1-D list of comments (not an np array)
"""
def createAndSaveTokenizer(train, maxWords):
  tokenizer = Tokenizer(num_words=maxWords)
  tokenizer.fit_on_texts(train)
  saveTokenizer(tokenizer, './preprocess/tokenizer.pickle')

"""
  Tokenize the data and pad each list of tokens to equal length.
  data - 1-D np-array or list of comments
"""
def createPaddedTokens(data, maxTokenLength, isTrainingFile):
  if(not fileExists('./preprocess/tokenizer.pickle') and not isTrainingFile):
    raise ValueError('Tokenizer has not been created/saved yet. Call createAndSaveTokenizer()')
  
  tokenizer = loadTokenizer('./preprocess/tokenizer.pickle')

  tokenizedData = tokenizer.texts_to_sequences(data)
  paddedTokenizedData = pad_sequences(tokenizedData, maxlen=maxTokenLength)
  return paddedTokenizedData