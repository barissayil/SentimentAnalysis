import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AlbertPreTrainedModel, AlbertModel, DistilBertPreTrainedModel, DistilBertModel

class BertForSentimentClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.bert = BertModel(config)
		#The classification layer that takes the [CLS] representation and outputs the logit
		self.cls_layer = nn.Linear(768, 1)

	def forward(self, seq, attn_masks):
		'''
		Inputs:
			-seq : Tensor of shape [B, T] containing token ids of sequences
			-attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the sequence length)
		'''
		#Feed the input to Bert model to obtain contextualized representations
		reps, _ = self.bert(seq, attention_mask = attn_masks)
		#Obtain the representations of [CLS] heads
		cls_reps = reps[:, 0]
		
		logits = self.cls_layer(cls_reps)
	
		return logits

class AlbertForSentimentClassification(AlbertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.albert = AlbertModel(config)
		#The classification layer that takes the [CLS] representation and outputs the logit
		self.cls_layer = nn.Linear(768, 1)

	def forward(self, seq, attn_masks):
		'''
		Inputs:
			-seq : Tensor of shape [B, T] containing token ids of sequences
			-attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the sequence length)
		'''
		#Feed the input to Albert model to obtain contextualized representations
		reps, _ = self.albert(seq, attention_mask = attn_masks)
		#Obtain the representations of [CLS] heads
		cls_reps = reps[:, 0]
		
		logits = self.cls_layer(cls_reps)
	
		return logits

class DistilBertForSentimentClassification(DistilBertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.distilbert = DistilBertModel(config)
		#The classification layer that takes the [CLS] representation and outputs the logit
		self.cls_layer = nn.Linear(768, 1)

	def forward(self, seq, attn_masks):
		'''
		Inputs:
			-seq : Tensor of shape [B, T] containing token ids of sequences
			-attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the sequence length)
		'''
		#Feed the input to DistilBert model to obtain contextualized representations
		reps, = self.distilbert(seq, attention_mask = attn_masks)
		#Obtain the representations of [CLS] heads
		cls_reps = reps[:, 0]
		
		logits = self.cls_layer(cls_reps)
	
		return logits