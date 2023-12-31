import torch
import torch.nn as nn

class NeuralNet(nn.Module): # feed forward neural net
  def __init__(self, input_size, hidden_size, num_classes): 
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.l2 = nn.Linear(hidden_size, hidden_size)
    self.l3 = nn.Linear(hidden_size, num_classes)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
    
  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.dropout(out)
    
    out = self.l2(out)
    out = self.relu(out)
    out = self.dropout(out)
    
    out = self.l3(out)
    out = self.relu(out)
    
    # no activation and no softmax because of future cross entropy loss
    
    return out
    
    
 
      