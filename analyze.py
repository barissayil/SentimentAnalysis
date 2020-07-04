import torch
from transformers import AutoTokenizer
from model import SentimentClassifier
from arguments import args

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
			print('Positive with probability {}%.'.format(int(prob*100)))
		else:
			print('Negative with probability {}%.'.format(int(100-prob*100)))

if __name__ == "__main__":
	print('Please wait while the analyser is being prepared.')
	#Create the model with the desired transformer model
	model = SentimentClassifier(model_name=args.model_name)
	#CPU or GPU
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#Put the model to the GPU if available
	model = model.to(device)
	#Load the state dictionary of the model
	model.load_state_dict(torch.load(f'models/{args.model_name}', map_location=device))
	#Initialize the tokenizer for the desired transformer model
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)
	sentence = input('Input sentiment to analyze: ')
	while sentence:
		classify_sentiment(sentence)
		sentence = input('Input sentiment to analyze: ')