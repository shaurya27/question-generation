import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
from constant import *

USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print ('USE GPU')
else:
    print ('USE CPU')

class EncoderParagraph(nn.Module):
    
    def __init__(self,word_size,word_dim, hidden_size, pretrained_word_embeds=None):
        super(EncoderParagraph, self).__init__()
        
        self.word_size = word_size
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.pretrained_word_embeds = pretrained_word_embeds
        self.embedding = nn.Embedding(self.word_size,self.word_dim,padding_idx=0)
        self.lstm = nn.LSTM(self.word_dim,self.hidden_size,batch_first = True,bidirectional=True)
	self._init_weights()
        
    def forward(self,x):
        out = self.embedding(x)
        out,hidden_cell = self.lstm(out)
        return out,hidden_cell

    def _init_weights(self):
        if PRE_TRAINED_EMBEDDING:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_word_embeds))
            if NON_TRAINABLE:
                self.embedding.weight.requires_grad = False
            else:
                self.embedding.weight.requires_grad = True
        else:
            nn.init.xavier_uniform_(self.embedding.weight.data)
            
class EncoderSentence(nn.Module):
    
    def __init__(self,word_size,word_dim, hidden_size, pretrained_word_embeds=None):
        super(EncoderSentence, self).__init__()
        
        self.word_size = word_size
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.pretrained_word_embeds = pretrained_word_embeds
        self.embedding = nn.Embedding(self.word_size,self.word_dim,padding_idx=0)
        self.lstm = nn.LSTM(self.word_dim,self.hidden_size,batch_first = True,bidirectional=True)
	self._init_weights()
        
    def forward(self,x):
        out = self.embedding(x)
        out,hidden_cell = self.lstm(out)
        return out,hidden_cell

    def _init_weights(self):
        if PRE_TRAINED_EMBEDDING:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_word_embeds))
            if NON_TRAINABLE:
                self.embedding.weight.requires_grad = False
            else:
                self.embedding.weight.requires_grad = True
        else:
            nn.init.xavier_uniform_(self.embedding.weight.data)
            
class AttnDecoderLSTM(nn.Module):
    def __init__(self, word_size,word_dim, hidden_size,max_length,pretrained_word_embeds=None):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.word_size = word_size
        self.word_dim = word_dim
        self.encoder_hidden_dim = hidden_size
        self.max_length = max_length
        
        self.embedding = nn.Embedding(self.word_size,self.word_dim,padding_idx=0)
        self.attn = nn.Linear(self.word_dim+self.encoder_hidden_dim, self.max_length)
        self.attn_combine = nn.Linear(self.word_dim+(self.encoder_hidden_dim/2), self.word_dim)
        #self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.word_dim,self.hidden_size,batch_first = True)
        self.out = nn.Linear(self.hidden_size, self.word_size)
	self._init_weights()

    def forward(self, input, hidden, encoder_output1):
        embedded = self.embedding(input)
        #print self.encoder_hidden_dim
        #print self.word_dim
        #print embedded.squeeze(1).size(),hidden[0].squeeze(0).size()
        #print torch.cat((embedded.squeeze(1),hidden[0].squeeze(0)),1).size()
        #print self.attn(torch.cat((embedded.squeeze(1),hidden[0].squeeze(0)),1)).size()
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded.squeeze(1),hidden[0].squeeze(0)),1)), dim=1)
        
        attn_weights = attn_weights.unsqueeze(1)
        
        # Apply Attention weights
        #print attn_weights.size(),encoder_output1.size()
        attn_applied = torch.bmm(attn_weights, encoder_output1)
        attn_applied = attn_applied.squeeze(1)
        #print attn_applied.size()
        # Prepare LSTM input tensor
        attn_combined = torch.cat((embedded.squeeze(1), attn_applied), 1)
        #print attn_combined.size()
        #print self.attn_combine(attn_combined).size()
        lstm_input = F.relu(self.attn_combine(attn_combined))
        lstm_input = lstm_input.unsqueeze(1)
        output, hidden = self.lstm(lstm_input, hidden)
        output = F.log_softmax(self.out(output[:,0,:]), dim=1)

        return output, hidden, attn_weights
    
    def _init_weights(self):
        if PRE_TRAINED_EMBEDDING:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_word_embeds))
            if NON_TRAINABLE:
                self.embedding.weight.requires_grad = False
            else:
                self.embedding.weight.requires_grad = True
        else:
            nn.init.xavier_uniform_(self.embedding.weight.data)
    
class QuestionGeneration(nn.Module):
    def __init__(self, para_encoder,sent_encoder, decoder):
        super(QuestionGeneration, self).__init__()
        
        self.encoder1 = para_encoder
        self.encoder2 = sent_encoder
        self.decoder = decoder 
        
        
    def forward(self, para_src,sent_src, trg, teacher_forcing_ratio=0.5):
        
        #src = [batch size, sent len]
        #trg = [batch size, sent len]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.word_size
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size,max_len, trg_vocab_size)
        if USE_GPU:
            outputs = outputs.cuda()
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        out1, hidden_cell1 = self.encoder1(para_src)
        out2, hidden_cell2 = self.encoder2(sent_src)
        hidden1 = (hidden_cell1[0].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell1[0].transpose(0,1).size()[0],-1)
        hidden2 = (hidden_cell2[0].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell1[0].transpose(0,1).size()[0],-1)
        hidden = torch.cat((hidden1,hidden2),dim=1).unsqueeze(0)
        cell1 = (hidden_cell1[1].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell1[0].transpose(0,1).size()[0],-1)
        cell2 = (hidden_cell2[1].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell1[0].transpose(0,1).size()[0],-1)
        cell = torch.cat((cell1,cell2),dim=1).unsqueeze(0)
        #cell = torch.cat((hidden_cell1[1][0::2],hidden_cell2[1][0::2]),dim=2)
        
        #first input to the decoder is the <sos> tokens
        input = trg[:,0]
        
        for t in range(1, max_len):
            input = input.unsqueeze(1)
            output, (hidden, cell), att_wt = self.decoder(input, (hidden, cell),out2)
            outputs[:,t,:] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[:,t] if teacher_force else top1)
        
        return outputs
