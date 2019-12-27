from flask import Flask, jsonify, request
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from model import SentimentClassifier

# Instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
# Enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})
#Create the network
net = SentimentClassifier()
#CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Put the network to the GPU if available
net = net.to(device)
#Instantiate the bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#Load the state dictionary of the network
net.load_state_dict(torch.load('./models/model', map_location=device))

def classify_sentiment(sentence):
	with torch.no_grad():
		tokens = tokenizer.tokenize(sentence)
		tokens = ['[CLS]'] + tokens + ['[SEP]']
		tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
		seq = torch.tensor(tokens_ids)
		seq = seq.unsqueeze(0)
		attn_mask = (seq != 0).long()
		logit = net(seq, attn_mask)
		prob = torch.sigmoid(logit.unsqueeze(-1))
		prob = prob.item()
		soft_prob = prob > 0.5
		if soft_prob == 1:
			return 'Positive', int(prob*100)
		else:
			return 'Negative', int(100-prob*100)

@app.route('/', methods=['GET'])
def sentiment():
	if request.method == 'GET':
		text = request.args['text']
		sentiment, probability = classify_sentiment(text)
		return jsonify({'sentiment': sentiment, 'probability': probability})

if __name__ == '__main__':
	app.run()