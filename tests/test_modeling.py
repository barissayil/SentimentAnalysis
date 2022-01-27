import torch.nn as nn
from transformers import BertPreTrainedModel, AlbertPreTrainedModel, DistilBertPreTrainedModel

from modeling import BertForSentimentClassification, AlbertForSentimentClassification, DistilBertForSentimentClassification

def test_bert():
	model = BertForSentimentClassification.from_pretrained('bert-base-uncased')
	assert isinstance(model, BertForSentimentClassification)
	assert isinstance(model, BertPreTrainedModel)
	assert isinstance(model, nn.Module)

def test_albert():
	model = AlbertForSentimentClassification.from_pretrained('albert-base-v2')
	assert isinstance(model, AlbertForSentimentClassification)
	assert isinstance(model, AlbertPreTrainedModel)
	assert isinstance(model, nn.Module)

def test_distilbert():
	model = DistilBertForSentimentClassification.from_pretrained('distilbert-base-uncased')
	assert isinstance(model, DistilBertForSentimentClassification)
	assert isinstance(model, DistilBertPreTrainedModel)
	assert isinstance(model, nn.Module)