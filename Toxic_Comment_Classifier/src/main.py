from time import time
from embedding.wordEmbeddings import createEmbeddingsFromCorpus, processPretrainedEmbeddings, loadSavedEmbeddings

def main():
  startTime = time()
  # Create embeddings from scratch
  # corpusFileName = "data/train.csv"
  # createEmbeddingsFromCorpus(corpusFileName)

  # Load pre-trained embeddings
  embeddingsFileName = "data/glove.6B.50d"
  # If embeddings has never been saved, save them.
  #processPretrainedEmbeddings(embeddingsFileName)
  # Re-use and load the saved embeddings
  embeddings = loadSavedEmbeddings(embeddingsFileName)

  print('Done! Completed in', time() - startTime, 'seconds')

if __name__ == "__main__":
  main()