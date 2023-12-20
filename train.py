

import json
import numpy as np
import random
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet


# create training data
with open('intents.json', 'r') as f:
  intents = json.load(f)
  
  
all_words = []

tags = []

xy = []

for intent in intents['intents']:
  tag = intent['tag']
  tags.append(tag)
  
  for pattern in intent['patterns']:
    w = tokenize(pattern)
    all_words.extend(w)
    xy.append((w, tag))
    
ignore_words = ['?', '!', '.', ',']

a = all_words

all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))


x_train =  []
y_train = []

for (pattern_sentence, tag) in xy:
  bag = bag_of_words(pattern_sentence, all_words)
  x_train.append(bag)
  label = tags.index(tag)
  y_train.append(label) # crossEntropyLoss
  
x_train = np.array(x_train)
y_train = np.array(y_train)
  
# pytorch model and training
class ChatDataset(Dataset):
  def __init__(self):
    self.n_samples = len(x_train)
    self.x_data = x_train
    self.y_data = y_train
    
  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]
  
  def __len__(self):
    return self.n_samples
  
batch_size = 4
hidden_size = 32
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.00001
num_epochs = 100000

dataset = ChatDataset()

train_loader = DataLoader(dataset = dataset, batch_size=batch_size, shuffle= True, num_workers=0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = NeuralNet(input_size= input_size, hidden_size= hidden_size, num_classes=  output_size).to(device)

# loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



 # Early stopping parameters
min_loss = np.inf
patience = 150
trigger_times = 0

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


            # Early stopping
            if min_loss > loss.item():
                min_loss = loss.item()
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print('Early stopping!')
                    break

print(f'final loss: {loss.item():.4f}')
# save and load model and implement chat

data = {
  "model_state" : model.state_dict(),
  "input_size" : input_size,
  "output_size" : output_size,
  "hidden_size" : hidden_size,
  "all_words" : all_words,
  "tags" : tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. fle saved to {FILE}')