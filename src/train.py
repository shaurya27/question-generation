from load_data import *
from model import *

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(tqdm.tqdm(iterator,desc='Training Processing')):
        
        src1 = batch[0]
        src2 = batch[1]
        trg = batch[2]
        src1 = Variable(src1.type(torch.LongTensor))
        src2 = Variable(src2.type(torch.LongTensor))
        trg = Variable(trg.type(torch.LongTensor))
        if USE_GPU:
            src1 = src1.cuda()
            src2 = src2.cuda()
            trg = trg.cuda()
        
        optimizer.zero_grad()
        
        output = model(src1,src2, trg)
        
        #trg = [batch size,sent len]
        #output = [batch size,sent len, output dim]
        
        #reshape to:
        #trg = [(sent len - 1) * batch size]
        #output = [(sent len - 1) * batch size, output dim]
        
        loss = criterion(output[:,1:,:].contiguous().view(-1, output.shape[2]), trg[:,1:].contiguous().view(-1))
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / iterator.dataset.length,optimizer

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(tqdm.tqdm(iterator,desc='validation Processing')):

            src1 = batch[0]
            src2 = batch[1]
            trg = batch[2]
            src1 = Variable(src1.type(torch.LongTensor))
            src2 = Variable(src2.type(torch.LongTensor))
            trg = Variable(trg.type(torch.LongTensor))
            if USE_GPU:
                src1 = src1.cuda()
                src2 = src2.cuda()
                trg = trg.cuda()

            output = model(src1,src2, trg, 0) #turn off teacher forcing

            loss = criterion(output[:,1:,:].contiguous().view(-1, output.shape[2]), trg[:,1:].contiguous().view(-1))

            epoch_loss += loss.item()
        
    return epoch_loss / iterator.dataset.length


def save(filename):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'valid_loss': valid_loss}
    torch.save(state, filename)
    
WORD_SIZE = len(word_mapping)

enc1 = EncoderParagraph(WORD_SIZE,WORD_DIM,HIDDEN_SIZE,pretrained_word_embeds)
enc2 = EncoderSentence(WORD_SIZE,WORD_DIM,HIDDEN_SIZE,pretrained_word_embeds)
dec = AttnDecoderLSTM(WORD_SIZE,WORD_DIM,HIDDEN_SIZE*4,MAX_SENT_LEN,pretrained_word_embeds)
model = QuestionGeneration(enc1,enc2, dec)

if USE_GPU:
    enc1 = enc1.cuda()
    enc2 = enc2.cuda()
    dec = dec.cuda()
    model = model.cuda()
    
    
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#optimizer = optim.Adam(model_parameters,LEARNING_RATE)
optimizer = optim.SGD(model_parameters,LEARNING_RATE,momentum= MOMENTUM)
init_lr = LEARNING_RATE

pad_idx = word_mapping['<pad>']
#criterion = nn.NLLLoss()
criterion = nn.NLLLoss(ignore_index=pad_idx)


#### Training

best_valid_loss = float('inf')

#if not os.path.isdir(f'{SAVE_DIR}'):
#    os.makedirs(f'{SAVE_DIR}')

val_dec_count = 0
for epoch in range(N_EPOCHS):
    
    train_loss,opti = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_dataloader, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        #save('../models/checkpoint_epoch_'+str(epoch)+'_valid_loss_'+str(valid_loss)+'_'+'.pth.tar')
        save('../models/best_validation_loss_model.pth.tar')
        val_dec_count = 0
    else:
        val_dec_count +=1
    save('../models/current_model.pth.tar')
        
    if val_dec_count >5:
        checkpoint = torch.load('../models/best_validation_loss_model.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        init_lr = init_lr/4.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        val_dec_count = 0
        
    
    print('Epoch [{}/{}] Train Loss: {:.4f} | Val. Loss: {:.4f}'.format(epoch+1, N_EPOCHS, train_loss,valid_loss))
