import torch
import torch.nn as nn
from transformers import BertModel

class SentimentClassifier(nn.Module):
    def __init__(self, freeze_bert = True):
        super(SentimentClassifier, self).__init__()
        #Instantiate BERT model object as the BERT layer of the classifier
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        #Freeze BERT layers if needed
        if freeze_bert:
            #For all the paramaters of the BERT layer
            for p in self.bert_layer.parameters():
                #Don't track their gradient since we won't modify them
                p.requires_grad = False
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
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)
        #Obtain the representation of [CLS] head
        cls_rep = cont_reps[:, 0]
        #Feed cls_rep to the classifier layer to get logits
        logits = self.cls_layer(cls_rep)
        #Return logits
        return logits