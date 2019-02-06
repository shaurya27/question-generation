import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *

class EncoderSentence(nn.Module):
    
    def __init__(self,word_size,word_dim, hidden_size, pretrained_word_embeds=None, output_type = 'sum'):
        super(EncoderSentence, self).__init__()
        
        self.output_type = output_type
        self.word_size = word_size
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.pretrained_word_embeds = pretrained_word_embeds
        self.embedding = nn.Embedding(self.word_size,self.word_dim,padding_idx=0)
        self.lstm = nn.LSTM(self.word_dim,self.hidden_size,batch_first = True,bidirectional=True)
        self._init_weights()
        
    def forward(self,x,input_lengths):
        embedded = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths,batch_first=True)
        outputs, hidden_cell = self.lstm(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        if self.output_type == 'sum':
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        elif self.output_type =='concat':
            outputs = torch.cat((outputs[:, :, :self.hidden_size], outputs[:, : ,self.hidden_size:]),dim=2)
        else:
            raise NotImplementedError 
        return outputs,hidden_cell

    def _init_weights(self):
        if PRE_TRAINED_EMBEDDING or WORD2VEC_EMBEDDING :
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_word_embeds))
            if NON_TRAINABLE:
                self.embedding.weight.requires_grad = False
            else:
                self.embedding.weight.requires_grad = True
        else:
            nn.init.xavier_uniform_(self.embedding.weight.data)