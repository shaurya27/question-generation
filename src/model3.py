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
        
    def forward(self,x,input_lengths):
        out = self.embedding(x)
        #print out.size()
        packed = torch.nn.utils.rnn.pack_padded_sequence(out, input_lengths,batch_first=True)
        out, hidden_cell = self.lstm(packed)
        # Unpack padding
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out,batch_first=True)
        #print out.size()
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
            

class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """
    def __init__(self,hidden_size, method="general"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':
            self.Wa = nn.Linear(hidden_size, (hidden_size/2), bias=False)
            #self.Wa = nn.Linear(hidden_size, (hidden_size/4), bias=False)
        elif method == "concat":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        elif method == 'bahdanau':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        else:
            raise NotImplementedError

    def forward(self, last_hidden, encoder_outputs):
        batch_size, seq_lens, _ = encoder_outputs.size()

        attention_energies = self._score(last_hidden, encoder_outputs, self.method)

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`, `bahdanau`)
        :return: a score (batch_size, max_time)
        """

        #assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1)
            x = F.tanh(self.Wa(torch.cat((x, encoder_outputs), 1)))
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)

        elif method == "bahdanau":
            x = last_hidden.unsqueeze(1)
            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1)

        else:
            raise NotImplementedError
            

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

    def forward(self, input, hidden, encoder_output):
      embedded = self.embedding(input)
      embedded = self.embedding_dropout(embedded)
        #embedded = embedded.unsqueeze(1)
      lstm_output,hidden = self.lstm(embedded,hidden)
      lstm_output = lstm_output.squeeze(1)
      attention_weights = self.attention.forward(lstm_output,encoder_output)
      context = attention_weights.unsqueeze(1).bmm(encoder_output).squeeze(1)
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
            

class QuestionGeneration(nn.Module):
    def __init__(self,sent_encoder, decoder):
        super(QuestionGeneration, self).__init__()
        
        #self.encoder1 = para_encoder
        self.encoder2 = sent_encoder
        self.decoder = decoder
        self.hidden_size = sent_encoder.hidden_size
        
        
    def forward(self,sent_src, trg,batch_len ,teacher_forcing_ratio=0.5):
        
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
        #out1, hidden_cell1 = self.encoder1(para_src)
        out2, hidden_cell2 = self.encoder2(sent_src,batch_len)
        #out1_ = out1[:,:,:self.hidden_size] + out1[:,:,self.hidden_size:]
        out2 = out2[:,:,:self.hidden_size] + out2[:,:,self.hidden_size:]
	#print out1_.size(),out2_.size()
        #out = torch.cat((out1_,out2_),dim=-1)
        #hidden1 = (hidden_cell1[0].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell1[0].transpose(0,1).size()[0],-1)
        hidden2 = (hidden_cell2[0].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell2[0].transpose(0,1).size()[0],-1)
        hidden = hidden2.unsqueeze(0)
        #cell1 = (hidden_cell1[1].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell1[0].transpose(0,1).size()[0],-1)
        cell2 = (hidden_cell2[1].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell2[0].transpose(0,1).size()[0],-1)
        cell = cell2.unsqueeze(0)
        #cell = torch.cat((hidden_cell1[1][0::2],hidden_cell2[1][0::2]),dim=2)
        #print hidden.size(),cell.size()
        #first input to the decoder is the <sos> tokens
        input = trg[:,0]
        
#        for t in range(1, max_len):
#            input = input.unsqueeze(1)
#            output, (hidden, cell), att_wt = self.decoder(input, (hidden, cell),out2)
#            outputs[:,t,:] = output
#            teacher_force = random.random() < teacher_forcing_ratio
#            top1 = output.max(1)[1]
#            input = (trg[:,t] if teacher_force else top1)
	teacher_force = random.random() < teacher_forcing_ratio
        
        if teacher_force:
          for t in range(1, max_len):
              input = input.unsqueeze(1)
              output, (hidden, cell), att_wt = self.decoder(input, (hidden, cell),out2)
              outputs[:,t,:] = output
          
              top1 = output.max(1)[1]
              input = trg[:,t]
        else:
          for t in range(1, max_len):
              input = input.unsqueeze(1)
              output, (hidden, cell), att_wt = self.decoder(input, (hidden, cell),out2)
              outputs[:,t,:] = output
              top1 = output.max(1)[1]
              input = top1        


        return outputs

