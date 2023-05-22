import math
import torch
from torch import nn, Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SinCosTextEncoder(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, init_range, device = device):
        super().__init__()
        
        # define the encoder
        self.encoder = nn.Embedding(n_tokens, d_model).to(device)

        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.d_model = d_model

    def forward(self, src: Tensor):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.encoder(src) * math.sqrt(self.d_model)


class SinCosPosEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000, device = device):
        super().__init__()

        self.positional_encoding = torch.zeros(max_seq_len, d_model).to(device)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        self.positional_encoding[:, 0::2] = torch.sin(positions)
        self.positional_encoding[:, 1::2] = torch.cos(positions)
        self.positional_encoding = self.positional_encoding.unsqueeze(1)
        
        self.register_buffer('pos_encoding', self.positional_encoding)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.positional_encoding[:x.size(0)]
        return self.dropout(x)
