from layers.encoder import Encoder
import torch
from torch import nn
import torch.nn.functional as F


class SentiNa(nn.Module):
    def __init__(self, vocab_size, num_encoder, num_heads, model_dim, max_len):
        super().__init__()


        #scale the weights from word2vec
        weights = torch.load("weights/embedding_weights.pt", weights_only=True)
        eps = 1e-8
        mu = weights.mean()
        sigma = weights.std()
        self.ln_embed = nn.LayerNorm(model_dim)
        # Scale to a standard deviation of 0.1 (common for Transformer stability)
        normalized_weights = (weights - mu) / (sigma + eps) * 0.02


        self.embedding = nn.Embedding.from_pretrained(normalized_weights, freeze=False, padding_idx=0)

        self.pos_embedding = nn.Embedding(max_len, model_dim)
        self.encoders = nn.ModuleList([Encoder(num_heads, model_dim) for i in range(num_encoder)])
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 2)
        )
        self.dropout = nn.Dropout(0.1)

        #scaling
        # nn.init.normal_(self.embedding.weight, 0, 0.02)
        nn.init.normal_(self.pos_embedding.weight, 0, 0.02)
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

        x = self.ln_embed(x)

        

        x = self.dropout(x)

        for encoder in self.encoders:
            x = encoder(x, attention_mask)
        
        #final shape of the output is (B, T, model_dim)
        #average across tokens

        mask = attention_mask.unsqueeze(-1).float()

        x = x * mask

        sum_x = torch.sum(x, dim = 1)

        token_counts = torch.clamp(mask.sum(dim=-2), min=1e-9)

        x = sum_x/token_counts

        x = self.classifier(x)

        return x
