import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from argparse import ArgumentParser
from model import SentimentClassifier
from dataset import SSTDataset

def get_accuracy_from_logits(logits, labels):
	#Get probabilities that the sentiments is positive by passing the logits through a sigmoid
	probs = torch.sigmoid(logits.unsqueeze(-1))
	#Convert probabilities to predictions, 1 being positive and 0 being negative
	preds = (probs > 0.5).long()
	#Check which predictions are the same as the ground truth and calculate the accuracy
	acc = (preds.squeeze() == labels).float().mean()
	#Return the accuracy
	return acc

def evaluate(net, criterion, dataloader):
	#Set net to evaluation mode
	net.eval()
	#Set mean accuracy, mean loss, and count to zero
	mean_acc, mean_loss, count = 0, 0, 0
	#We won't track the gradients since we're evaluating the model, not training it
	with torch.no_grad():
		#Get the sequence, attention masks, and labels from the dataloader
		for seq, attn_masks, labels in dataloader:
			#Put the sequence, attention masks, and labels to the GPU, if available
			seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
			#Get the logits from the network
			logits = net(seq, attn_masks)
			#Calculate the mean loss using logits and labels
			mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
			#Calculate the man accuracy using logits and labels
			mean_acc += get_accuracy_from_logits(logits, labels)
			#Increment the count
			count += 1
	#Return accuracy and loss
	return mean_acc / count, mean_loss / count

def train(net, criterion, optimizer, train_loader, val_loader, args):
	#Set best accuracy to zero
	best_acc = 0
	for epoch in range(args.num_eps):
		for i, (seq, attn_masks, labels) in enumerate(train_loader):
			#Clear gradients
			optimizer.zero_grad()  
			#Convert these to cuda tensors a.k.a. put them to GPU (if available)
			seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
			#Obtain the logits from the model
			logits = net(seq, attn_masks)
			#Compute loss loss
			loss = criterion(logits.squeeze(-1), labels.float())
			#Backpropagate the gradients
			loss.backward()
			#Optimization step
			optimizer.step()
			#Display the loss and accuracy at fixed intervals
			if (i + 1) % args.print_every == 0:
				#Get the accuracy for the current batch using logits and labels
				acc = get_accuracy_from_logits(logits, labels)
				#Display the accuracy
				print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(i, epoch, loss.item(), acc))
		#Calculate the validation accuracy and loss
		val_acc, val_loss = evaluate(net, criterion, val_loader)
		#Display the validation accuracy and loss
		print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(epoch, val_acc, val_loss))
		#If the validation accuracy of the current network is the best one yet
		if val_acc > best_acc:
			#Print the old best validation accuracy and the current one
			print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
			#Set the current validation accuracy as the best one
			best_acc = val_acc
			#Save the current network's state dictionary
			torch.save(net.state_dict(), 'models/model')

if __name__ == "__main__":
	#Get the parameters from arguments if used
	parser = ArgumentParser()
	parser.add_argument('-freeze_bert', action='store_true')
	parser.add_argument('-maxlen', type = int, default= 25)
	parser.add_argument('-batch_size', type = int, default= 32)
	parser.add_argument('-lr', type = float, default = 2e-5)
	parser.add_argument('-print_every', type = int, default= 100)
	parser.add_argument('-num_eps', type = int, default= 5)
	args = parser.parse_args()
	#Instantiate the classifier model
	net = SentimentClassifier(args.freeze_bert)
	#CPU or GPU
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#Put the network to the GPU if available
	net = net.to(device)
	#Takes as the input the logits of the positive class and computes the binary cross-entropy 
	criterion = nn.BCEWithLogitsLoss()
	#Adam optimizer
	optimizer = optim.Adam(net.parameters(), lr = args.lr)
	#Create instances of training and validation set
	train_set = SSTDataset(filename = 'data/train.tsv', maxlen = args.maxlen)
	val_set = SSTDataset(filename = 'data/dev.tsv', maxlen = args.maxlen)
	#Create intsances of training and validation dataloaders
	train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 5)
	val_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = 5)
	#Train the network
	train(net, criterion, optimizer, train_loader, val_loader, args)