import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

class SSTDataset(Dataset):
	"""
	Stanford Sentiment Treebank V1.0
	Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
	Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
	Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
	"""
	def __init__(self, filename, maxlen): 
		#Store the contents of the file in a pandas dataframe
		self.df = pd.read_csv(filename, delimiter = '\t')
		#Print the filename and the dataframe to take a look
		print(filename)
		print(self.df)
		print()
		#Initialize the BERT tokenizer
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		#Maximum length of the tokens list to keep all the sequences of fixed size
		self.maxlen = maxlen

	def __len__(self):
		#Return the number of data points present in the dataset
		return len(self.df)

	def __getitem__(self, index):    
		#Select the sentence and label at the specified index in the data frame
		sentence = self.df.loc[index, 'sentence']
		label = self.df.loc[index, 'label']
		#Preprocess the text to be suitable for BERT
		#Tokenize the sentence
		tokens = self.tokenizer.tokenize(sentence) 
		#Insert the CLS and SEP token in the beginning and end of the sentence
		tokens = ['[CLS]'] + tokens + ['[SEP]'] 
		#Check if tokens list is smaller than or bigger than (or equal to) the maximum allowed size
		if len(tokens) < self.maxlen:
			#Padd sentences
			tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 
		else:
			#Prun the list to be of specified max length
			tokens = tokens[:self.maxlen-1] + ['[SEP]'] 
		#Obtain the indices of the tokens in the BERT Vocabulary
		tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
		#Convert the list to a pytorch tensor
		seq = torch.tensor(tokens_ids) 
		#Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
		attn_mask = (seq != 0).long()
		#Return sequence, attention mask, and label
		return seq, attn_mask, label