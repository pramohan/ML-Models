import math
import torch
from torch import nn, Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IndexTextEncoder(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, init_range, device = device):
        super().__init__()

        # define the encoder
        self.encoder = nn.Embedding(n_tokens, d_model - 1).to(device)

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


class IndexPosEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000, device = device):
        super().__init__()

        pos = torch.arange(0, max_seq_len, dtype=torch.float).to(device) / max_seq_len
        self.positional_encoding = pos.unsqueeze(1).unsqueeze(1)
        
        self.register_buffer('pos_encoding', self.positional_encoding)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        # concatenate ``positional_encoding`` to x
        pe = self.positional_encoding.expand(-1, x.size(1), -1)
        x = torch.cat((pe[:x.size(0)], x), dim=2)

        return self.dropout(x)
