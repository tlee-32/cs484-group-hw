from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd

def main():
    #Preprocessing data
    trainData = pd.read_csv("./data/train.csv")
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

if __name__ == "__main__":
    main()