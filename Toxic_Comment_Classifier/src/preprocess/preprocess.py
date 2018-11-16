def getVocabSize(sentences):
  corpus = []
  # Add each word to the corpus
  for sentence in sentences:
    for word in sentence:
      corpus.append(word)
  # Remove duplicates
  corpus = set(corpus)

  return len(corpus)