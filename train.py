import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer
from modeling import BertForSentimentClassification, AlbertForSentimentClassification, DistilBertForSentimentClassification
from dataset import SSTDataset
from evaluate import evaluate
from arguments import args

def train(model, criterion, optimizer, train_loader, val_loader, args):
	best_acc = 0
	for epoch in trange(args.num_eps, desc="Epoch"):
		model.train()
		for i, (input_ids, attention_mask, labels) in enumerate(tqdm(iterable=train_loader, desc="Training")):
			optimizer.zero_grad()  
			input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
			logits = model(input_ids=input_ids, attention_mask=attention_mask)
			loss = criterion(input=logits.squeeze(-1), target=labels.float())
			loss.backward()
			optimizer.step()
		val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
		print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(epoch, val_acc, val_loss))
		if val_acc > best_acc:
			print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
			best_acc = val_acc
			model.save_pretrained(save_directory=f'models/{args.output_dir}/')
			config.save_pretrained(save_directory=f'models/{args.output_dir}/')
			tokenizer.save_pretrained(save_directory=f'models/{args.output_dir}/')

if __name__ == "__main__":

	if args.model_name_or_path is None:
		args.model_name_or_path = 'bert-base-uncased'

	#Configuration for the desired transformer model
	config = AutoConfig.from_pretrained(args.model_name_or_path)
	
	#Tokenizer for the desired transformer model
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	
	#Create the model with the desired transformer model
	if config.model_type == 'bert':
		model = BertForSentimentClassification.from_pretrained(args.model_name_or_path, config=config)
	elif config.model_type == 'albert':
		model = AlbertForSentimentClassification.from_pretrained(args.model_name_or_path, config=config)
	elif config.model_type == 'distilbert':
		model = DistilBertForSentimentClassification.from_pretrained(args.model_name_or_path, config=config)
	else:
		raise ValueError('This transformer model is not supported yet.')
		
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	#Takes as the input the logits of the positive class and computes the binary cross-entropy 
	criterion = nn.BCEWithLogitsLoss()

	optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

	train_set = SSTDataset(filename='data/train.tsv', maxlen=args.maxlen_train, tokenizer=tokenizer)
	val_set = SSTDataset(filename='data/dev.tsv', maxlen=args.maxlen_val, tokenizer=tokenizer)

	train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_threads)
	val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads)

	train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, args=args)
