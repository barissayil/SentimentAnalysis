import argparse

import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SSTDataset
from analyzer import Analyzer

args = argparse.Namespace()
args.model_name_or_path = 'barissayil/bert-sentiment-analysis-sst'
args.output_dir = 'my_model'
args.maxlen_val = 50
args.batch_size = 32
args.num_threads = 1

def test_evaluate():
	analyzer = Analyzer(will_train=False, args=args)
	criterion = nn.BCEWithLogitsLoss()
	val_set = SSTDataset(filename='data/dev.tsv', maxlen=args.maxlen_val, tokenizer=analyzer.tokenizer)
	val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads)
	val_accuracy, val_loss = analyzer.evaluate(val_loader=val_loader, criterion=criterion)
	
	assert round(val_accuracy * 100) == 92
	assert round(val_loss) == 6