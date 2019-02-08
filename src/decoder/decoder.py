import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *
from attention.attention import Attention

class AttnDecoderLSTM(nn.Module):
    def __init__(self, word_size,word_dim, hidden_size,pretrained_word_embeds=None):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.word_size = word_size
        self.word_dim = word_dim
        self.embedding_dropout = nn.Dropout(0.3)
        self.pretrained_word_embeds = pretrained_word_embeds
        self.embedding = nn.Embedding(self.word_size,self.word_dim,padding_idx=0)
        self.lstm = nn.LSTM(self.word_dim,self.hidden_size,batch_first = True)
        self.attention = Attention(self.hidden_size)
        #self.character_distribution = nn.Linear(5*(self.hidden_size/4), self.word_size)
        self.character_distribution = nn.Linear(3*(self.hidden_size/2), self.word_size)
        self._init_weights()
        
        
        #self.out = nn.Linear(self.hidden_size, self.word_size)

    def forward(self, input, hidden, encoder_output,encoder_batch_len):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        #embedded = embedded.unsqueeze(1)
        lstm_output,hidden = self.lstm(embedded,hidden)
        lstm_output = lstm_output.squeeze(1)
        attention_weights = self.attention.forward(lstm_output,encoder_output,encoder_batch_len)
        #print attention_weights.size(), encoder_output.size()
        context = attention_weights.unsqueeze(1).bmm(encoder_output).squeeze(1)
        #print context.size(),lstm_output.size()
        #print "hello"
        #print context.size(),lstm_output.size(),torch.cat((lstm_output, context), 1).size()
        output = self.character_distribution(torch.cat((lstm_output, context), 1))
        output = F.log_softmax(output, dim=1)

        return output, hidden, attention_weights
    
    def _init_weights(self):
        if PRE_TRAINED_EMBEDDING:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_word_embeds))
            if NON_TRAINABLE:
                self.embedding.weight.requires_grad = False
            else:
                self.embedding.weight.requires_grad = True
        else:
            nn.init.xavier_uniform_(self.embedding.weight.data)