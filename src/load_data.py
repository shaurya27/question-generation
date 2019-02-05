import numpy as np
import json
import tqdm 
import string
import codecs

from constant import *
import pickle

pretrained_word_embeds = np.load("../data/pre_trained_embedding.npy")
word_mapping = pickle.load( open( "../data/word_mapping.p", "rb" ) )

MAX_SENT_LEN = 90
MAX_PARA_LEN = 350

def data_generator(filepath):
    with open(filepath,'r') as f:
        for line in f:
            yield json.loads(line)
        return
####################################
### Dataloader
####################################
class CustomDataset():

    def __init__(self,file_path,length,word2idx):
        self.file_path = file_path
        self.length = length
        self.word2idx = word2idx
        self.gen = data_generator(self.file_path)

    def __getitem__(self,index):
        try:
            text = self.gen.next()
        except StopIteration:
            self.gen = data_generator(self.file_path)
            text = self.gen.next()
        paragraph = text['paragraph'].split()
        paragraph.insert(0, "<start>")
        paragraph.append('<end>')
        sentence = text['sentence'].split()
        sentence.insert(0, "<start>")
        sentence.append('<end>')
        question = text['question'].split()
        question.insert(0, "<start>")
        question.append('<end>')
        
        x_paragraph = [self.word2idx.get(word) if self.word2idx.get(word) else self.word2idx['<unk>'] for word in paragraph]
        x_sentence = [self.word2idx.get(word) if self.word2idx.get(word) else self.word2idx['<unk>'] for word in sentence]
        x_question = [self.word2idx.get(word) if self.word2idx.get(word) else self.word2idx['<unk>'] for word in question]
        paragraph_word_len = len(paragraph)
        sentence_word_len = len(sentence)
        question_word_len = len(question)
        return {'paragraph': paragraph,'sentence': sentence,'question': question,
                'paragraph_word_id':x_paragraph,'sentence_word_id':x_sentence,'question_word_id':x_question,
                "paragraph_word_len": paragraph_word_len,"sentence_word_len": sentence_word_len,"question_word_len":question_word_len}
    
    def __len__(self):
        return self.length
    
def collate_fn(batch):
    #paragraph_max_word = [item['paragraph_word_len'] for item in batch]
    #sentence_max_word = [item['sentence_word_len'] for item in batch]
    question_max_word = [item['question_word_len'] for item in batch]
    paragraph_max_len = MAX_PARA_LEN
    sentence_max_len = MAX_SENT_LEN
    question_max_len = max(question_max_word)

    paragraph_word_data = np.zeros((len(batch),paragraph_max_len))
    sentence_word_data = np.zeros((len(batch),sentence_max_len))
    question_word_data = np.zeros((len(batch),question_max_len))
    for i,item in enumerate(batch):
        paragraph_word_data[i,:len(item['paragraph_word_id'][:MAX_PARA_LEN])] = item['paragraph_word_id'][:MAX_PARA_LEN]
        sentence_word_data[i,:len(item['sentence_word_id'][:MAX_SENT_LEN])] = item['sentence_word_id'][:MAX_SENT_LEN]
        question_word_data[i,:len(item['question_word_id'])] = item['question_word_id']
    paragraph =[item['paragraph'] for item in batch]
    sentence =[item['sentence'] for item in batch]
    question =[item['question'] for item in batch]
    return torch.tensor(paragraph_word_data),torch.tensor(sentence_word_data),torch.tensor(question_word_data),paragraph,sentence,question

import torch
from torch.utils.data import DataLoader

### Load datas
train_dataloader = DataLoader(CustomDataset(TRAIN_DATA_PATH,TRAIN_DATA_LENGTH,word_mapping),
                              batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=False)

valid_dataloader = DataLoader(CustomDataset(VALID_DATA_PATH,VALID_DATA_LENGTH,word_mapping),
                              batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=False)

test_dataloader = DataLoader(CustomDataset(TEST_DATA_PATH,TEST_DATA_LENGTH,word_mapping),
                              batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=False)