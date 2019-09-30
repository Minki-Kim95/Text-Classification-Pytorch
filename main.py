import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.CNN import CNN
from models.LSTM import LSTMClassifier
from models.LSTM_Attn import AttentionModel
from models.RCNN import RCNN
from models.RNN import RNN


def inputNumber(message):
    while True:
        try:
            userInput = int(input(message))
            if userInput < 0 or userInput >4:
                print("out of range")
                continue
        except ValueError:
            print("Not an integer! Try again.")
            continue
        else:
            return userInput
            break

# MAIN PROGRAM STARTS HERE:
choice_model = inputNumber("select Model(0: CNN, 1: LSTM, 2: LSTM_Attn, 3: RCNN, 4: RNN): ")
start_time = time.time()    # store start time

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()
print("load is end")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device setting is end")

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0

    # model.cuda()
    # Now send existing model to device.
    print("total epoch loss allocated")
    model = model.to(device)
    print("device allocated")
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    print("optim allocated")
    steps = 0
    model.train()
    print("model train end")
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        print("TEXT")
        target = batch.label
        print("batch label")
        target = torch.autograd.Variable(target).long()
        print("target allocated")
        # if torch.cuda.is_available():
        #     text = text.cuda()
        #     target = target.cuda()
        text = text.to(device)
        target = target.to(device)
        print("text and target allocated")

        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        print("optim zero grad")
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            # if torch.cuda.is_available():
            #     text = text.cuda()
            #     target = target.cuda()
            text = text.to(device)
            target = target.to(device)

            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
	

learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300

if choice_model == 0:
    in_channels = 1
    out_channels = 128
    kernel_heights = [3, 4, 5]
    stride = 1
    padding = 0
    keep_probab = 0.8
    model = CNN(batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, word_embeddings)
elif choice_model ==1:
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
elif choice_model == 2:
    model = AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
elif choice_model == 3:
    model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
elif choice_model == 4:
    model = RNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
else:
    print("No...")
print("model allocated")
loss_fn = F.cross_entropy
print("loss_fn allocated")
for epoch in range(10):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    print("train_loss allocated")
    val_loss, val_acc = eval_model(model, valid_iter)
    print("val_loss allocated")
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    
test_loss, test_acc = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

test_sen2 = TEXT.preprocess(test_sen2)
test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
test_tensor = Variable(test_sen, volatile=True)
# test_tensor = test_tensor.cuda()
test_tensor = test_tensor.to(device)
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print ("Sentiment: Positive")
else:
    print ("Sentiment: Negative")

end_time = time.time()  #store end time
print("WorkingTime: {} sec".format(end_time-start_time))