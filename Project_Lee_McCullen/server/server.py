from flask import Flask, request, jsonify
from keras.preprocessing import text, sequence
import tensorflow as tf

from model.cnn import KerasCNN
from preprocess.tokenutil import createPaddedTokens, tokenizeText, cleanText
from preprocess.fileutil import loadTokenizer
from analysis.plot import plotModel

MAX_TOKEN_LENGTH = 100

app = Flask(__name__)
model = None
graph = None
tokenizer = None

def start():
  global app
  app.run()

def loadModel(ml, modelFile):
  global model
  model = ml
  model.loadModel(modelFile)
  
  global graph
  graph = tf.get_default_graph()
  
  global tokenizer
  tokenizer = loadTokenizer('./preprocess/tokenizer.pickle')

@app.route("/predict", methods=["POST"])
def predict():
  res = {"success": False}
  
  # ensure an image was properly uploaded to our endpoint
  if request.method == 'POST':
    if request.json.get('text'):
      # read text
      rawText = [cleanText(request.json['text'])]
      
      tokenizedText = tokenizer.texts_to_sequences(rawText)
      paddedTokeniedText = sequence.pad_sequences(tokenizedText, maxlen=MAX_TOKEN_LENGTH)
      with graph.as_default():
        prediction = model.predict(paddedTokeniedText)

      labels = ['toxic', 'severely toxic', 'obscene', 'threat', 'insult', 'identity_hate']
      scores = dict(map(lambda k,v: (k,v), labels, prediction.tolist()[0]))
      res["prediction"] = scores

      # indicate that the request was a success
      res["success"] = True

  return jsonify(res)
