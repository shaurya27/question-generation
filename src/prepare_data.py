import numpy as np
import json
from tqdm import tqdm_notebook as tqdm
import string
import codecs
import pickle

from constant import *

def data_generator(filepath):
    with open(filepath,'r') as f:
        for line in f:
            yield json.loads(line)
        return
    
def create_mapping(filepath,min_count):
    gen = data_generator(filepath)
    word_count = {}
    word_map = {}
    sent_lengths = []
    para_lengths = []
    for data in tqdm(gen):
        ques = data['question'].split()
        paragraph = data['paragraph'].split()
        sentence = data['sentence'].split()
        para_lengths.append(len(paragraph))
        sent_lengths.append(len(sentence))
        text = ques + paragraph
        for item in text:
            if word_count.get(item):
                word_count[item]+=1
            else:
                word_count[item] = 1
    for k,v in word_count.iteritems():
        if v>min_count:
            word_map[k] = len(word_map)+1
    word_map['<pad>'] = 0
    word_map['<unk>'] = len(word_map)
    word_map['<start>'] = len(word_map)
    word_map['<end>'] = len(word_map)
    return word_map,para_lengths,sent_lengths

word_mapping,para_len,sent_len = create_mapping('../data/processed/train_data.json',5)

print "maximum sentence length : {}".format(max(sent_len))
print "minimum sentence length : {}".format(min(sent_len))
print "mean sentence length : {}".format(np.mean(sent_len))
print "std dev sentence length : {}".format(np.std(sent_len))
print "mean + 3*std_dev sentence length : {}".format(np.mean(sent_len)+3*np.std(sent_len))
print "\n"
print "maximum paragraph length : {}".format(max(para_len))
print "minimum paragraph length : {}".format(min(para_len))
print "mean paragraph length : {}".format(np.mean(para_len))
print "std dev paragraph length : {}".format(np.std(para_len))
print "mean + 3*std_dev paragraph length : {}".format(np.mean(para_len)+3*np.std(para_len))

################################################
## pre trained embedding
#################################################
all_word_embeds = {}
for i, line in enumerate(codecs.open(PRE_TRAINED_EMBEDDING_PATH, 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == WORD_DIM + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

#Intializing Word Embedding Matrix
pretrained_word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_mapping), WORD_DIM))

for w in word_mapping:
    if w.lower() in all_word_embeds:
        pretrained_word_embeds[word_mapping[w]] = all_word_embeds[w.lower()]

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))
## To save memory
del all_word_embeds

np.save("../data/pre_trained_embedding.npy",pretrained_word_embeds)
pickle.dump(word_mapping,open( "../data/word_mapping.p", "wb" ))