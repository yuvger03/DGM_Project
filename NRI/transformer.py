import torch
import torch.nn as nn

class PairwiseAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: [batch, num_atoms, timesteps, features]
        B, N, T, F = x.shape
        x = x.view(B * N, T, F)           # [B*N, T, F]
        x = self.linear(x)                # project to embed_dim
        out, _ = self.attn(x, x, x)       # self-attention over time
        out = out.view(B, N, T, -1)       # reshape back
        return out
