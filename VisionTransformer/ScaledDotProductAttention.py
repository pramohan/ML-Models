import torch
import numpy as np
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    ''' Implements the Scaled Dot Product Attention function as defined in
        Attention Is All You Need (Vaswani et al., 2017).
        Args:
        - d_k (int): the dimension of the key vectors
    '''
    def __init__(self, d_k, dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.d_k = d_k

    def forward(self, Q, K, V):
        ''' Computes the scaled dot product attention :
            Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

            Args:
            - Q (torch.Tensor): the query tensor of shape (batch_size, num_queries, dim_k)
            - K (torch.Tensor): the key tensor of shape (batch_size, num_keys, dim_k)
            - V (torch.Tensor): the value tensor of shape (batch_size, num_keys, dim_v)

            Returns:
            - attention_output (torch.Tensor): the output tensor of shape (batch_size, num_queries, dim_v), obtained
            by applying the attention mechanism to the input values V using the input queries Q and keys K
            - attention_scores (torch.Tensor): the attention tensor of shape (batch_size, num_queries, num_keys),
            representing the attention scores between each query and key (output of softmax)
        '''
        attention = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        
        att_softmax = self.dropout(nn.functional.softmax(attention, dim=-1))
        att_out = torch.matmul(att_softmax, V)
        
        attention_output, attention_scores = att_out, att_softmax

        return attention_output, attention_scores
