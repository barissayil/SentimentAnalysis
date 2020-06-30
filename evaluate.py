import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from model import SentimentClassifier
from dataset import SSTDataset

#Create validation set
val_set = SSTDataset(filename = 'data/dev.tsv', maxlen = 30)
#Create validation dataloader
val_loader = DataLoader(val_set, batch_size = 64, num_workers = 5)
#Create the model
model = SentimentClassifier()
#CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Put the model to the GPU if available
model = model.to(device)
#Load the state dictionary of the model
model.load_state_dict(torch.load('./models/model', map_location=device))
#Takes as the input the logits of the positive class and computes the binary cross-entropy 
criterion = nn.BCEWithLogitsLoss()

def get_accuracy_from_logits(logits, labels):
	#Get a tensor of shape [B, 1, 1] with probabilities that the sentiment is positive
	probs = torch.sigmoid(logits.unsqueeze(-1))
	#Convert probabilities to predictions, 1 being positive and 0 being negative
	soft_probs = (probs > 0.5).long()
	#Check which predictions are the same as the ground truth and calculate the accuracy
	acc = (soft_probs.squeeze() == labels).float().mean()
	#Return the accuracy
	return acc

def evaluate(model, criterion, dataloader):
	#Set model to evaluation mode
	model.eval()
	#Set mean accuracy, mean loss, and count to zero
	mean_acc, mean_loss, count = 0, 0, 0
	#We won't track the gradients since we're evaluating the model, not training it
	with torch.no_grad():
		#Get the sequence, attention masks, and labels from the dataloader
		for seq, attn_masks, labels in tqdm(dataloader, desc="Evaluating"):
			#Put the sequence, attention masks, and labels to the GPU, if available
			seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
			#Get the logits from the model
			logits = model(seq, attn_masks)
			#Calculate the mean loss using logits and labels
			mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
			#Calculate the man accuracy using logits and labels
			mean_acc += get_accuracy_from_logits(logits, labels)
			#Increment the count
			count += 1
	#Return accuracy and loss
	return mean_acc / count, mean_loss / count

#Get validation accuracy and validation loss
val_acc, val_loss = evaluate(model, criterion, val_loader)
print("Validation Accuracy : {}, Validation Loss : {}".format(val_acc, val_loss))