import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AlbertPreTrainedModel, AlbertModel

class BertForSentimentClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		#Instantiate BERT model object as the BERT layer of the classifier
		self.bert = BertModel(config)
		#Instantiate the classification layer
		self.cls_layer = nn.Linear(768, 1)

	def forward(self, seq, attn_masks):
		'''
		Inputs:
			-seq : Tensor of shape [B, T] containing token ids of sequences
			-attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the sequence length)
		'''
		#Feed the input to BERT model to obtain contextualized representations
		reps, _ = self.bert(seq, attention_mask = attn_masks)
		#Obtain the representation of [CLS] head
		cls_rep = reps[:, 0]
		#Feed cls_rep to the classifier layer to get logits
		logits = self.cls_layer(cls_rep)
		#Return logits
		return logits

class AlbertForSentimentClassification(AlbertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		#Instantiate 	ALBERT model object as the BERT layer of the classifier
		self.albert = AlbertModel(config)
		#Instantiate the classification layer
		self.cls_layer = nn.Linear(768, 1)

	def forward(self, seq, attn_masks):
		'''
		Inputs:
			-seq : Tensor of shape [B, T] containing token ids of sequences
			-attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the sequence length)
		'''
		#Feed the input to ALBERT model to obtain contextualized representations
		reps, _ = self.albert(seq, attention_mask = attn_masks)
		#Obtain the representation of [CLS] head
		cls_rep = reps[:, 0]
		#Feed cls_rep to the classifier layer to get logits
		logits = self.cls_layer(cls_rep)
		#Return logits
		return logits
