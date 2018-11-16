from nltk import word_tokenize

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

