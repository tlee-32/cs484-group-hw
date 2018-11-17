from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 7740 instances of http
# 1654 instances of @
# Remove newlines, quotes, and punctuation
def cleanColumn(dfCol):
  dfCol = dfCol.replace(to_replace=r'http', value='', regex=True)
  dfCol = dfCol.replace(to_replace=r'@', value='at', regex=True) 
  dfCol = dfCol.replace(to_replace=r'[!#$%&()*+,-.\/:;<=>?[\]^_`{|}~\'\`\"\_\n\\]', value=' ', regex=True)  
  dfCol = dfCol.str.lower()
  return dfCol
  
"""
  Apply tokenize function on comments.
"""
def tokenizeComments(commentColumn):
  result = []
  comments = commentColumn.tolist()
  for comment in comments:
    result.append(word_tokenize(comment))
  return result

"""
  Tokenize the data and pad each list of tokens to equal length.
"""
def createPaddedTokens(data, maxWords, tokenLength):
  tokenizer = Tokenizer(num_words=maxWords)
  tokenizer.fit_on_texts(data)
  tokens = tokenizer.texts_to_sequences(data)
  paddedTokens = pad_sequences(tokens, maxlen=tokenLength)
  return paddedTokens
