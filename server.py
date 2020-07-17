from flask import Flask, jsonify, request
from flask_cors import CORS

import torch
from transformers import AutoTokenizer, AutoConfig
from modeling import BertForSentimentClassification, AlbertForSentimentClassification, DistilBertForSentimentClassification
from arguments import args

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

	if args.model_name_or_path is None:
		args.model_name_or_path = 'barissayil/bert-sentiment-analysis-sst'

	#Configuration for the desired transformer model
	config = AutoConfig.from_pretrained(args.model_name_or_path)

	#Create the model with the desired transformer model
	if config.model_type == 'bert':
		model = BertForSentimentClassification.from_pretrained(args.model_name_or_path)
	elif config.model_type == 'albert':
		model = AlbertForSentimentClassification.from_pretrained(args.model_name_or_path)
	elif config.model_type == 'distilbert':
		model = DistilBertForSentimentClassification.from_pretrained(args.model_name_or_path)
	else:
		raise ValueError('This transformer model is not supported yet.')

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	
	model.eval()

	#Initialize the tokenizer for the desired transformer model
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	
	#Run the Flask App
	app.run()