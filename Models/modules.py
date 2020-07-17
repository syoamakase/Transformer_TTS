import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(q, k, v, d_k, mask, dropout):
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e10)
        scores = torch.softmax(scores, dim=-1)
    else:
        scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = F.dropout(scores, dropout)
    output = torch.matmul(scores, v)
    return output, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1, concat_after=False):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = dropout
        self.concat_after = concat_after
        if self.concat_after:
            self.out = nn.Linear(2*d_model, d_model)
        else:
            self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, debug=False):
        bs = q.size(0)
        if self.concat_after:
            context_vector = q

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        #if debug:
        #    print(f'k {k.max()} {k.min()} {k.mean()}')
        #    print(f'q {q.max()} {q.min()} {q.mean()}')
        #    print(f'v {v.max()} {v.min()} {v.mean()}')
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores, attn = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        if self.concat_after:
            concat = torch.cat((context_vector, concat), dim=-1)
        output = self.out(concat)

        return output, attn

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_conv_kernel_size, dropout=0.1):
        super().__init__()

        self.f_1 = nn.Conv1d(d_model, d_model*4, kernel_size=ff_conv_kernel_size, padding=int(ff_conv_kernel_size/2))
        self.dropout = nn.Dropout(dropout)
        self.f_2 = nn.Conv1d(d_model*4, d_model, kernel_size=ff_conv_kernel_size, padding=int(ff_conv_kernel_size/2))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = F.relu(self.f_1(x.transpose(1,2)))
        x = self.f_2(x).transpose(1,2)
        x = x + res
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x

class PositionalEncoder(nn.Module):
    # copy
    def __init__(self, d_model, max_seq_len=1500, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.shape[1]
        pe = self.pe[:,:seq_len]
        if x.is_cuda:
            pe.cuda()
        x = x + self.alpha * pe
        return self.dropout(x)
