from gensim.summarization.summarizer import summarize
from load_data3 import *
from model3 import *

#def user_input(data,word2idx=word_mapping):
#    paragraph = data['user_paragraph'] 
#    x_paragraph = [word2idx.get(word) if word2idx.get(word) else word2idx['<unk>'] for word in paragraph]
#    x_paragraph.insert(0,word_mapping['<start>'])
#    x_paragraph.append(word_mapping['<end>'])
#    important_sentence = summarize(paragraph,ratio=0.8, word_count=None, split=True)
#    model_input_para = np.zeros((len(important_sentence),MAX_PARA_LEN))
#    model_input_sent = np.zeros((len(important_sentence),MAX_SENT_LEN))
#    for t,sentence in enumerate(important_sentence):
#        x_sentence = [word2idx.get(word) if word2idx.get(word) else word2idx['<unk>'] for word in sentence]
#        x_sentence.insert(0,word_mapping['<start>'])
#        x_sentence.append(word_mapping['<end>'])
#        model_input_para[t,:] =  x_paragraph[:MAX_PARA_LEN]
#        model_input_sent[t,:] =  x_sentence[:MAX_SENT_LEN]
#    return torch.tensor(model_input_para),torch.tensor(model_input_sent)

def user_input(data,word2idx=word_mapping):
    paragraph = data['user_paragraph'] 
    x_paragraph = [word2idx.get(word) if word2idx.get(word) else word2idx['<unk>'] for word in paragraph]
    x_paragraph.insert(0,word_mapping['<start>'])
    x_paragraph.append(word_mapping['<end>'])
    important_sentence = summarize(paragraph,ratio=0.8, word_count=None, split=True)
    model_input_para = np.zeros((len(important_sentence),MAX_PARA_LEN))
    #model_input_sent = np.zeros((len(important_sentence),MAX_SENT_LEN))
    sentences=[]
    batch_len=[]
    for t,sentence in enumerate(important_sentence):
        x_sentence = [word2idx.get(word) if word2idx.get(word) else word2idx['<unk>'] for word in sentence.split()]
        x_sentence.insert(0,word_mapping['<start>'])
        x_sentence.append(word_mapping['<end>'])
        model_input_para[t,:] =  x_paragraph[:MAX_PARA_LEN]
        #model_input_sent[t,:] =  x_sentence[:MAX_SENT_LEN]
        sentences.append(x_sentence)
        batch_len.append(len(x_sentence))
    sentences.sort(key= len, reverse=True)
    batch_len.sort(reverse=True)
    model_input_sent = pad(sentences)
    return torch.tensor(model_input_para),torch.tensor(model_input_sent),torch.tensor(batch_len)