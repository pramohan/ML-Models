import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention

class TransformerBlock(nn.Module):
    ''' Implements the Transformer Encoder block:
            [LayerNorm, Multi-head attention, residual,
             LayerNorm, MLP, residual]

    '''
    def __init__(self, input_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Define LayerNorm
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

        # Define MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, input_dim)
        )

        # Define MultiHeadAttention
        self.multihead_attention = MultiHeadAttention(input_dim, num_heads, dropout)

    def forward(self, x):
        res1 = x
        # Apply LayerNorm
        x = self.layer_norm1(x)
        # Define Q, K, V
        Q, K, V = x, x, x
        # Pass (Q, K, V) to MultiHeadAttention
        x, _ = self.multihead_attention(Q, K, V)
        # Sum with 1st residual connection
        x += res1
        # Apply LayerNorm
        res2 = x
        x = self.layer_norm2(x)
        # Pass to MLP
        x = self.mlp(x)
        # Sum with 2nd residual connection
        x += res2
        return x
