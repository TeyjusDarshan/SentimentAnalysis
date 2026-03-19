from torch import nn
import torch
import math

class AttentionHead(nn.Module):
    def __init__(self, model_embed_size, output_embed_size):
        super().__init__()

        self.qW = nn.Linear(model_embed_size, output_embed_size) #shape (C, Oc)
        self.kW = nn.Linear(model_embed_size, output_embed_size)
        self.vW = nn.Linear(model_embed_size, output_embed_size)
        self.output_embed_size = output_embed_size
        self.dropout = nn.Dropout(0.3)

        #scaling of weights
        nn.init.normal_(self.qW.weight, mean = 0.0, std = 0.02)
        nn.init.normal_(self.kW.weight, mean = 0.0, std = 0.02)
        nn.init.normal_(self.vW.weight, mean = 0.0, std = 0.02)

        #bias scaling
        nn.init.constant_(self.qW.bias, 0)
        nn.init.constant_(self.kW.bias, 0)
        nn.init.constant_(self.vW.bias, 0)


    
    def forward(self, x, attention_mask = None):
        #shape of x is (B, T, C)
        Q = self.qW(x) #shape (B, T, Oc)
        K = self.kW(x)
        V = self.vW(x) #shape (B, T, Oc)

        attention = Q @ K.transpose(-1, -2) / math.sqrt(self.output_embed_size)#shape (B, T, T)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)
            attention = attention.masked_fill(mask == 0, float('-inf'))
        

        attention_probs = torch.softmax(attention, dim = -1)
        attention_probs = self.dropout(attention_probs)

        output = attention_probs @ V #shape (B, T, Oc)
        
        return output








