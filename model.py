import torch
import torch.nn as nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
		def __init__(self, model_name='bert-base-uncased'):
				super(SentimentClassifier, self).__init__()
				#Instantiate BERT model object as the BERT layer of the classifier
				self.transformer_layer = AutoModel.from_pretrained(model_name)
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
				reps, _ = self.transformer_layer(seq, attention_mask = attn_masks)
				#Obtain the representation of [CLS] head
				cls_rep = reps[:, 0]
				#Feed cls_rep to the classifier layer to get logits
				logits = self.cls_layer(cls_rep)
				#Return logits
				return logits
