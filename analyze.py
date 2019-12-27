import torch
from transformers import BertTokenizer
from model import SentimentClassifier

print('Please wait while the analyser is being prepared.')

#Create the network with keeping bert layers unfrozen
net = SentimentClassifier(freeze_bert = False)
#CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Put the network to the GPU if available
net = net.to(device)
#Load the state dictionary of the network that I've trained
net.load_state_dict(torch.load('./models/model', map_location=device))
#Bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
			print('Positive with probability {}%.'.format(int(prob*100)))
		else:
			print('Negative with probability {}%.'.format(int(100-prob*100)))

if __name__ == "__main__":
	sentence = input('Input sentiment to analyze: ')
	while sentence:
		classify_sentiment(sentence)
		sentence = input('Input sentiment to analyze: ')