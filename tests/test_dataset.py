import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from dataset import SSTDataset

args = argparse.Namespace()
args.maxlen_train = 30
args.maxlen_val = 50
args.batch_size = 32
args.num_threads = 1

# Initialize tokenizer for BERT.
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def test_train_set():
	train_set = SSTDataset(filename='data/train.tsv', maxlen=args.maxlen_train, tokenizer=tokenizer)
	assert isinstance(train_set, SSTDataset)
	assert isinstance(train_set, Dataset)
	assert len(train_set) == 67349

	train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_threads)
	assert isinstance(train_loader, DataLoader)

	input_ids, attention_mask, labels = next(iter(train_loader))
	assert input_ids.size() == torch.Size([args.batch_size, args.maxlen_train])
	assert attention_mask.size() == torch.Size([args.batch_size, args.maxlen_train])
	assert labels.size() == torch.Size([args.batch_size])

  
def test_val_set():
	val_set = SSTDataset(filename='data/dev.tsv', maxlen=args.maxlen_val, tokenizer=tokenizer)
	assert isinstance(val_set, SSTDataset)
	assert isinstance(val_set, Dataset)
	assert len(val_set) == 872

	val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads)
	assert isinstance(val_loader, DataLoader)

	input_ids, attention_mask, labels = next(iter(val_loader))
	assert input_ids.size() == torch.Size([args.batch_size, args.maxlen_val])
	assert attention_mask.size() == torch.Size([args.batch_size, args.maxlen_val])
	assert labels.size() == torch.Size([args.batch_size])