from layers.encoder import Encoder
import torch
from torch import nn


class SentiNa(nn.Module):
    def __init__(self, vocab_size, num_encoder, num_heads, model_dim, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(max_len, model_dim)
        self.encoders = nn.ModuleList([Encoder(num_heads, model_dim) for i in range(num_encoder)])
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim//2), 
            nn.ReLU(),
            nn.Linear(model_dim//2, 1)
        )

        #scaling
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.02)
                nn.init.constant_(layer.bias, 0)

    def forward(self, token_ids, attention_mask):
        #shape of token_ids is (B, T)
        B, T = token_ids.shape
        x = self.embedding(token_ids) #shape (B, T, model_dim)

        positions = torch.arange(T, device = token_ids.device)
        pos_enc = self.pos_embedding(positions)

        x = x + pos_enc

        for encoder in self.encoders:
            x = encoder(x)
        
        #final shape of the output is (B, T, model_dim)
        #average across tokens

        x = torch.mean(x, dim = -2) # output (B, model_dim)

        x = self.classifier(x)

        return x
