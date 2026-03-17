from torch import nn
import torch
import math

class AttentionHead(nn.Module):
    def __init__(self, model_embed_size, output_embed_size, attention_mask = None):
        super().__init__()

        self.qW = nn.Linear(model_embed_size, output_embed_size) #shape (C, Oc)
        self.kW = nn.Linear(model_embed_size, output_embed_size)
        self.vW = nn.Linear(model_embed_size, output_embed_size)
        self.output_embed_size = output_embed_size
        self.mask = attention_mask

        #scaling of weights
        nn.init.normal_(self.qW.weight, mean = 0.0, std = 0.02)
        nn.init.normal_(self.kW.weight, mean = 0.0, std = 0.02)
        nn.init.normal_(self.vW.weight, mean = 0.0, std = 0.02)

        #bias scaling
        nn.init.constant_(self.qW.bias, 0)
        nn.init.constant_(self.kW.bias, 0)
        nn.init.constant_(self.vW.bias, 0)


    
    def forward(self, x):
        #shape of x is (B, T, C)
        Q = self.qW(x) #shape (B, T, Oc)
        K = self.kW(x)
        V = self.vW(x) #shape (B, T, Oc)

        attention = Q @ K.transpose(-1, -2) / math.sqrt(self.output_embed_size)#shape (B, T, T)
        if self.mask is not None:
            mask = self.mask.unsqueeze(1)
            attention = attention.masked_fill(mask == 0, float('-inf'))

        output = attention @ V #shape (B, T, Oc)
        
        return output








