import pandas as pd
import torch
from torch.utils.data import Dataset

class SSTDataset(Dataset):
	"""
	Stanford Sentiment Treebank V1.0
	Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
	Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
	Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
	"""
	def __init__(self, filename, maxlen, tokenizer): 
		# Store the contents of the file in pandas dataframe.
		self.df = pd.read_csv(filename, delimiter = '\t')
		# Initialize tokenizer for the desired transformer model.
		self.tokenizer = tokenizer
		# Maximum length of tokens list to keep all the sequences of fixed size.
		self.maxlen = maxlen

	def __len__(self):
		# Return length of dataframe.
		return len(self.df)

	def __getitem__(self, index):    
		# Select sentence and label at specified index from data frame.
		sentence = self.df.loc[index, 'sentence']
		label = self.df.loc[index, 'label']
		# Preprocess text to be suitable for transformer
		tokens = self.tokenizer.tokenize(sentence) 
		tokens = ['[CLS]'] + tokens + ['[SEP]'] 
		if len(tokens) < self.maxlen:
			tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 
		else:
			tokens = tokens[:self.maxlen-1] + ['[SEP]'] 
		# Obtain indices of tokens and convert them to tensor.
		input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens)) 
		# Obtain attention mask i.e. a tensor containing 1s for no padded tokens and 0s for padded ones.
		attention_mask = (input_ids != 0).long()
		# Return input IDs, attention mask, and label.
		return input_ids, attention_mask, label
