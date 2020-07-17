import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig

class SSTDataset(Dataset):
	"""
	Stanford Sentiment Treebank V1.0
	Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
	Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
	Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
	"""
	def __init__(self, filename, maxlen, tokenizer): 
		#Store the contents of the file in a pandas dataframe
		self.df = pd.read_csv(filename, delimiter = '\t')
		#Initialize the tokenizer for the desired transformer model
		self.tokenizer = tokenizer
		#Maximum length of the tokens list to keep all the sequences of fixed size
		self.maxlen = maxlen

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):    
		#Select the sentence and label at the specified index in the data frame
		sentence = self.df.loc[index, 'sentence']
		label = self.df.loc[index, 'label']
		#Preprocess the text to be suitable for the transformer
		tokens = self.tokenizer.tokenize(sentence) 
		tokens = ['[CLS]'] + tokens + ['[SEP]'] 
		if len(tokens) < self.maxlen:
			tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 
		else:
			tokens = tokens[:self.maxlen-1] + ['[SEP]'] 
		#Obtain the indices of the tokens in the BERT Vocabulary
		input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
		input_ids = torch.tensor(input_ids) 
		#Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
		attention_mask = (input_ids != 0).long()
		return input_ids, attention_mask, label
