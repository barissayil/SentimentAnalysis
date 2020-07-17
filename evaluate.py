import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from modeling import BertForSentimentClassification, AlbertForSentimentClassification, DistilBertForSentimentClassification
from dataset import SSTDataset
from arguments import args


def get_accuracy_from_logits(logits, labels):
	#Get a tensor of shape [B, 1, 1] with probabilities that the sentiment is positive
	probs = torch.sigmoid(logits.unsqueeze(-1))
	#Convert probabilities to predictions, 1 being positive and 0 being negative
	soft_probs = (probs > 0.5).long()
	#Check which predictions are the same as the ground truth and calculate the accuracy
	acc = (soft_probs.squeeze() == labels).float().mean()
	return acc

def evaluate(model, criterion, dataloader, device):
	model.eval()
	mean_acc, mean_loss, count = 0, 0, 0
	with torch.no_grad():
		for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
			input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
			logits = model(input_ids, attention_mask)
			mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
			mean_acc += get_accuracy_from_logits(logits, labels)
			count += 1
	return mean_acc / count, mean_loss / count

if __name__ == "__main__":

	if args.model_name_or_path is None:
		args.model_name_or_path = 'barissayil/bert-sentiment-analysis-sst'

	#Configuration for the desired transformer model
	config = AutoConfig.from_pretrained(args.model_name_or_path)

	#Tokenizer for the desired transformer model
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	
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

	#Takes as the input the logits of the positive class and computes the binary cross-entropy 
	criterion = nn.BCEWithLogitsLoss()

	val_set = SSTDataset(filename='data/dev.tsv', maxlen=args.maxlen_val, tokenizer=tokenizer)
	val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads)
	
	val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
	print("Validation Accuracy : {}, Validation Loss : {}".format(val_acc, val_loss))
