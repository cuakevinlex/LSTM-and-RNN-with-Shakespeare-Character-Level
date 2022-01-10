#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))


# In[2]:


import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import io


# In[3]:


# Hyperparameters
BATCH_SIZE=100
input_size = 32
hidden_size = 128
num_layers = 2
sequence_length = 50
learning_rate = 0.00001
batch_size = 64
num_epochs = 10


# In[4]:


data_URL = "shakespeare_train.txt"
with io.open(data_URL, 'r', encoding='utf-8') as f:
    text = f.read()
# Characters' Collection
char = set(text)
# construct character dictionary
char_to_int = {c : i for i , c in enumerate( char )}
int_to_char = dict(enumerate(char))
# Encode data , shape = [ number of c h a r a ct e r s ]

num_classes = len(int_to_char)

dtype = torch.FloatTensor
training_data = np.array([char_to_int[c] for c in text], dtype=np.int32)


# In[5]:


data_URL = "shakespeare_valid.txt"
with io.open(data_URL, 'r', encoding='utf-8') as f:
    text = f.read()
# Characters' Collection
char_valid = set(text)

val_data = np.array([char_to_int[c] for c in text], dtype=np.int32)


# In[6]:


print(char)
#{'S', 'O', 'e', 'v', 'P', 'w', '$', 'u', 'Y', 'C', '3', 'A', 'Q', 'j', 'T', 'V', 'q', 'f', 'p', 'F', 'U', 'Z', 'y', ':', '.', 'r', ' ', 'x', 'm', 'g', 'J', 'L', 'a', ',', 'H', 'M', 'G', ']', 'n', 'W', 'c', "'", 'k', 'I', '\n', 'd', 'o', 'h', 'l', 'R', '&', 'B', 'N', 'b', 'z', 's', 'i', 't', '?', 'D', '-', ';', 'K', 'E', 'X', '[', '!'}


# In[7]:


def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot


# In[8]:


text = ""
for i in training_data:
    text+= str(int_to_char[i])
#print(text)


# In[9]:


text = ""
for i in val_data:
    text+= str(int_to_char[i])
print(text)


# In[10]:


def build_sequences(text, window):
    x = list()
    y = list()
    
    for i in range(len(text)):
        try:
            # Get window of chars from text then, transform it into its idx representation
            sequence = text[i:i+window].tolist()
            
            # Get word target
            target = text[i+window]
            
            # Save sequences and targets
            x.append(sequence)
            y.append(target)
            
        except Exception as e: 
            pass
        
    x = np.array(x)
    y = np.array(y)
    
    return x, y


# In[11]:


# Creates data based on text array and sets x as the first sequence count target is char right after
x, y = build_sequences(training_data, sequence_length)
x_val, y_val = build_sequences(val_data, sequence_length)


# In[12]:


print(x)
print(y)


# In[13]:


x = Variable(torch.from_numpy(x))
y = Variable(torch.from_numpy(y))
x_val = Variable(torch.from_numpy(x_val))
y_val = Variable(torch.from_numpy(y_val))


# In[14]:


training_data = torch.utils.data.TensorDataset(x,y.long())
val_data = torch.utils.data.TensorDataset(x_val, y_val.long())


train_count = len(training_data)
val_count = len(val_data)

# training_data, val_data = torch.utils.data.random_split(training_data, [int(train_count*.9), train_count - int(train_count*.9)])
train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)

#used for generating data as seed
sequences = []
for batch_idx, (data, targets) in enumerate(train_loader):
    for seq in data:
        sequences.append(seq.numpy())
sequences = np.array(sequences)


# In[15]:


# storing gpu availability on boolean var
is_cuda = torch.cuda.is_available()

# checking for GPU
if is_cuda:
    device = torch.device("cuda")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


# In[16]:


class LSTM(nn.Module):
    def __init__(self, sequence_length, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        self.lstm = nn.LSTM(num_classes, hidden_size, num_layers, batch_first=True)
        self.fully_connected = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
       
        # Forward propagate LSTM
        x, _ = self.lstm(x, (h0,c0))
        x = x.reshape(x.shape[0], -1)

        # Decode the hidden state of the last time step
        # print(out.shape)
        x = self.fully_connected(x)
        
        return x


# In[17]:


# Initialize network
lstm = LSTM(sequence_length, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)


# In[18]:


learning_curve = []
training_acc = []
val_acc = []
# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        
        # Get data to cuda if possible
        data = one_hot_encode(data, num_classes)
        data = torch.from_numpy(data).to(device=device)
        targets = targets.to(device=device)
        
        # forward
        scores = lstm(data)
        loss = criterion(scores, targets)
        cel = loss.item()
        
        # accuracy and loss
        predicted = torch.max(scores.data, 1)[1] 
        correct = (predicted == targets).sum()
        accuracy = correct/len(predicted)*100
        training_acc.append(accuracy.item())
        learning_curve.append(cel)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        
        # validation computation
        x_val, y_val = next(iter(val_loader))
        x_val = one_hot_encode(x_val,num_classes)
        x_val = torch.from_numpy(x_val).to(device=device)
        y_val = y_val.to(device=device)
        score_val = lstm(x_val)
        val_loss = criterion(score_val, y_val)
        val_cel = val_loss.item()
        
        # val acciracu
        predicted_val = torch.max(score_val.data, 1)[1] 
        correct_val = (predicted_val == y_val).sum()
        accuracy_val = correct_val/len(predicted_val)*100
        val_acc.append(accuracy_val.item())
        
        # gradient descent or adam step
        optimizer.step()
        if batch_idx % 250 == 0: 
            currently_run = batch_idx*len(data)
            data_total = len(train_loader.dataset)
            percent_of_data_run = 100.*batch_idx / len(train_loader)
            print('Epoch: {}[{}/{} ({:.0f}%)]\t Error:{:.2f}\t Accuracy:{:.2f}\t Val Error:{:.2f}'.format(
              epoch, currently_run, data_total, percent_of_data_run, cel, accuracy, val_cel))
    PATH = 'lstm_' + str(epoch) + 'epochs.pt'
    torch.save(lstm.state_dict(), PATH)


# In[19]:


histogram = plt.figure()
histogram.set_size_inches(15, 5)
# training accuracy
plt.subplot(1,2,1)
plt.plot(training_acc, label='training accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Accuracy Rate")
plt.title("Accuracy")

# learning curve
plt.subplot(1,2,2)
plt.plot(learning_curve, label='Cross Entropy Loss')
plt.legend()
plt.xlabel("Loss")
plt.ylabel("Iteration")
plt.title("Learning Curve")

chart_title = 'LSTM: hidden size: ' + str(hidden_size) + ', num_layers: ' + str(num_layers) + ', sequence_length: ' + str(sequence_length)
title = 'LSTM.hidden_size.' + str(hidden_size) + '.num_layers.' + str(num_layers) + '.sequence_length.' + str(sequence_length)+".png"
histogram.suptitle(chart_title)
histogram.savefig(title)


# In[20]:


def generator(model, sequences, idx_to_char, vocab_to_int, n_chars):
    text = ""
    # Set the model in evalulation mode
    #model.eval()

    softmax = nn.Softmax(dim=1)
    
    # select seed randomly from dataset
    #start = np.random.randint(0, len(sequences)-1)
    
    # use first entry as seed
    start = 0
    
    # The pattern is defined given the random idx
    pattern = sequences[start]
    
    # print using dictionaries to convert number to text
    
    # save to text file
    text += "\nPattern: \n"
    pat_text = ""
    for value in pattern:
        pat_text += idx_to_char[value]
    text += pat_text
    
    # print
    print("\nPattern: \n")
    print(''.join([idx_to_char[value] for value in pattern]), "\"")
    
    # In full_prediction we will save the complete prediction
    full_prediction = pattern.copy()
   
    # The prediction starts, it is going to be predicted a given
    # number of characters
    for i in range(n_chars):
        
        # The numpy patterns is transformed into a tesor-type and reshaped
        pattern_one_hot = one_hot_encode(np.array([pattern],dtype=int), num_classes)
        pattern_one_hot = torch.from_numpy(pattern_one_hot).type(torch.float).to(device)
        prediction = model(pattern_one_hot)
        prediction = softmax(prediction)
        prediction = prediction.squeeze().detach().cpu().numpy()
        arg_max = np.argmax(prediction)
        pattern = pattern[1:]
        pattern = np.append(pattern, arg_max)
        
        # The full prediction is saved
        full_prediction = np.append(full_prediction, arg_max)
    text += "\nPrediction: \n"
    pred_text = ""
    for value in full_prediction:
        pred_text += idx_to_char[value]
    text += pred_text
    print("Prediction: \n")
    print(''.join([idx_to_char[value] for value in full_prediction]), "\"")
    return text


# In[21]:


text_generated = "Hidden size: " + str(hidden_size) +", sequence length: " +str(sequence_length) +"\n"
for epoch in range(num_epochs):
    model_lstm = LSTM(sequence_length, hidden_size, num_layers, num_classes).to(device)
    PATH = 'lstm_' + str(epoch) + 'epochs.pt'
    model_lstm.load_state_dict(torch.load(PATH))
    model_lstm.eval()
    text_generated += "\nEpoch: " + str(epoch+1) + "\n"
    text_generated += generator(model_lstm, sequences, int_to_char, char_to_int, 2500) + "\n"


# In[ ]:


# PATH = 'lstm_5epochs.pt'
# torch.save(lstm.state_dict(), PATH)
with open('text generation_128h50seq.txt', 'w') as w:
    w.write(str(text_generated))


# In[47]:


def generator_fixed(model, sequences, idx_to_char, vocab_to_int, n_chars):
    text = ""
    # Set the model in evalulation mode
    #model.eval()

    softmax = nn.Softmax(dim=1)
    
    seed = "CORDELIA:\nHad you not been their father, these whi"
    
    # print using dictionaries to convert number to text
    pattern = np.array([char_to_int[c] for c in seed], dtype=np.int32)
    
    # save to text file
    text += "\nPattern: \n"
    pat_text = ""
    for value in pattern:
        pat_text += idx_to_char[value]
    text += pat_text
    
    # print
    print("\nPattern: \n")
    print(''.join([idx_to_char[value] for value in pattern]), "\"")
    
    # In full_prediction we will save the complete prediction
    full_prediction = pattern.copy()
   
    # The prediction starts, it is going to be predicted a given
    # number of characters
    for i in range(n_chars):
        
        # The numpy patterns is transformed into a tesor-type and reshaped
        pattern_one_hot = one_hot_encode(np.array([pattern],dtype=int), num_classes)
        pattern_one_hot = torch.from_numpy(pattern_one_hot).type(torch.float).to(device)
        prediction = model(pattern_one_hot)
        prediction = softmax(prediction)
        
        prediction = prediction.squeeze().detach().cpu().numpy()
        arg_max = np.argmax(prediction)
        pattern = pattern[1:]
        pattern = np.append(pattern, arg_max)
        
        # The full prediction is saved
        full_prediction = np.append(full_prediction, arg_max)
    text += "\nPrediction: \n"
    pred_text = ""
    for value in full_prediction:
        pred_text += idx_to_char[value]
    text += pred_text
    print("Prediction: \n")
    print(''.join([idx_to_char[value] for value in full_prediction]), "\"")
    return text


# In[49]:


text_generated = "Hidden size: " + str(hidden_size) +", sequence length: " +str(sequence_length) +"\n"
for epoch in range(num_epochs):
    model_lstm = LSTM(sequence_length, hidden_size, num_layers, num_classes).to(device)
    PATH = 'lstm_' + str(epoch) + 'epochs.pt'
    model_lstm.load_state_dict(torch.load(PATH))
    model_lstm.eval()
    text_generated += "\nEpoch: " + str(epoch+1) + "\n"
    text_generated += generator_fixed(model_lstm, sequences, int_to_char, char_to_int, 2500) + "\n"


# In[50]:


# PATH = 'lstm_5epochs.pt'
# torch.save(lstm.state_dict(), PATH)
with open('text generation_128h50seq_fixed_seed.txt', 'w') as w:
    w.write(str(text_generated))


# In[ ]:




