from user_input import *

WORD_SIZE = len(word_mapping)

enc1 = EncoderParagraph(WORD_SIZE,WORD_DIM,HIDDEN_SIZE,pretrained_word_embeds)
enc2 = EncoderSentence(WORD_SIZE,WORD_DIM,HIDDEN_SIZE,pretrained_word_embeds)
dec = AttnDecoderLSTM(WORD_SIZE,WORD_DIM,HIDDEN_SIZE*4,MAX_SENT_LEN,pretrained_word_embeds)
model = QuestionGeneration(enc1,enc2, dec)

checkpoint = torch.load('../models/best_validation_loss_model.pth.tar',map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])

encoder1 = model.encoder1
encoder2 = model.encoder2
decoder = model.decoder

id2word = {}
for k,v in word_mapping.iteritems():
    id2word[v]=k
    
def give_results(out):
    result = []
    for j in out:
        res = []
        for k in j:
            if id2word[k] == '<end>':
                break
            else:
                res.append(id2word[k])
        result.append(" ".join(res))
    return result

def evaluate(input_tensor1, input_tensor2,encoder1,encoder2,decoder):
    with torch.no_grad():
        encoder1.eval()
        encoder2.eval()
        decoder.eval()
        input_tensor1 = Variable(input_tensor1.type(torch.LongTensor))
        input_tensor2 = Variable(input_tensor2.type(torch.LongTensor))
        
        out1, hidden_cell1 = encoder1(input_tensor1)
        out2, hidden_cell2 = encoder2(input_tensor2)
        #hidden = torch.cat((hidden_cell1[0][0::2],hidden_cell2[0][0::2]),dim=2)
        #cell = torch.cat((hidden_cell1[1][0::2],hidden_cell2[1][0::2]),dim=2)
        hidden1 = (hidden_cell1[0].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell1[0].transpose(0,1).size()[0],-1)
        hidden2 = (hidden_cell2[0].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell1[0].transpose(0,1).size()[0],-1)
        hidden = torch.cat((hidden1,hidden2),dim=1).unsqueeze(0)
        cell1 = (hidden_cell1[1].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell1[0].transpose(0,1).size()[0],-1)
        cell2 = (hidden_cell2[1].transpose(0,1).transpose(1,2)).contiguous().view(hidden_cell1[0].transpose(0,1).size()[0],-1)
        cell = torch.cat((cell1,cell2),dim=1).unsqueeze(0)
        
        output_list=[]
        input = input_tensor1[:,0]
        input = input.unsqueeze(1)
        outputs = np.zeros((input_tensor1.size(0),20))
        for i in range(20):
            #print input.size()
            output, (hidden, cell),at_wt = decoder(input, (hidden, cell),out2)
            top1 = output.max(1)[1]
            output_list.append(top1)
            outputs[:,i] = top1
            input = top1.unsqueeze(1)
        output_ = give_results(outputs)
        return output_
    
