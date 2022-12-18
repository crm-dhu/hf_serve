from flask import Flask, request, jsonify
from transformers import pipeline

# load the model bert-base-uncased
unmasker = pipeline('fill-mask', model='bert-base-uncased')

app = Flask(__name__)
input_key = "input"

@app.route("/")
def hello():
  return "use endpoint /unmask and provide a masked_str with [MASK] filling the token being masked."

@app.route("/unmask", methods=["POST"])
def unmask():
  masked_str = request.json[input_key]
  return unmasked(masked_str)

def unmasked(x: str): 
  y = unmasker(x)
  y.sort(key=lambda i: i['score'], reverse=True)
  return jsonify(input = x, output = y)  

def init():
  print("init called to load model for bert-base-uncased")
  return unmasked("[MASK] is a music instrument.")

with app.app_context():
  init()