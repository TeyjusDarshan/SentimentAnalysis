from layers.multihead_attention import *
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.mlh = MultiheadAttention(num_heads, model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, 4*model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(0.1)
        )

        self.ln2 = nn.LayerNorm(model_dim)

        #scaling
        for layer in self.ff:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.02)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, attention_mask = None):
        x = self.mlh(x, attention_mask)

        ln_res = self.ln2(x)
        y = self.ff(ln_res)

        return x + y

    
