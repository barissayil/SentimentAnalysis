from flask import Flask, jsonify, request
from flask_cors import CORS

import torch
from transformers import AutoTokenizer
from model import SentimentClassifier
from arguments import args

# Instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
# Enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

def classify_sentiment(sentence):
	with torch.no_grad():
		tokens = tokenizer.tokenize(sentence)
		tokens = ['[CLS]'] + tokens + ['[SEP]']
		tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
		seq = torch.tensor(tokens_ids)
		seq = seq.unsqueeze(0)
		attn_mask = (seq != 0).long()
		logit = model(seq, attn_mask)
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
	#Create the model with the desired transformer model
	model = SentimentClassifier(model_name=args.model_name)
	#CPU or GPU
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#Put the model to the GPU if available
	model = model.to(device)
	#Initialize the tokenizer for the desired transformer model
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)
	#Load the state dictionary of the model
	model.load_state_dict(torch.load(f'models/{args.model_name}', map_location=device))
	app.run()