from layers.attention_head import AttentionHead
import torch
from torch import nn

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, model_embed_size):
        super().__init__()
        self.num_heads = num_heads
        self.model_embed_size = model_embed_size
        self.attention_heads = nn.ModuleList([AttentionHead(model_embed_size, model_embed_size//num_heads) for i in range(num_heads)])
        self.projection_layer = nn.Linear(model_embed_size, model_embed_size)
        self.ln = nn.LayerNorm(model_embed_size)
        self.dropout = nn.Dropout(0.3)

        #scaling 
        nn.init.normal_(self.projection_layer.weight, 0, 0.02)
        nn.init.constant_(self.projection_layer.bias, 0)
    
    def forward(self, x, attention_mask = None):
        norm_x = self.ln(x)
        head_outputs = [h(norm_x, attention_mask) for h in self.attention_heads]
        out = torch.cat(head_outputs, dim = -1) #shape (B, T, model_embed_size)
        projection = self.projection_layer(out)
        projection = self.dropout(projection)
        residual_output =  x + projection
        return residual_output



