import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(q, k, v, d_k, mask, dropout, debug=False):
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        #scores = scores.masked_fill(mask==0, -1e10)
        scores = scores.masked_fill(mask==0, -1e4)
        scores = torch.softmax(scores, dim=-1)
    else:
        scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = F.dropout(scores, dropout)
    output = torch.matmul(scores, v)
    return output, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, q_dim, k_dim, v_dim, d_model, dropout=0.1, concat_after=False):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads

        self.h = heads

        self.q_linear = nn.Linear(q_dim, d_model)
        self.v_linear = nn.Linear(k_dim, d_model)
        self.k_linear = nn.Linear(v_dim, d_model)

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
        scores, attn = attention(q, k, v, self.d_k, mask, self.dropout, debug)
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
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
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
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.shape[1]
        pe = self.pe[:,:seq_len].to(x.device)
        x = x + self.alpha * pe
        return self.dropout(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        causal = False
        kernel_size = 31
        padding = self.calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1)
        self.depth_conv1 = DepthwiseConv(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B x T x H) -> (B x H x T)

        x = self.layer_norm(x).transpose(1,2)
        x = self.pointwise_conv1(x)
        out, gate = x.chunk(2, dim=1)
        x = out * gate.sigmoid()
        x = self.depth_conv1(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x).transpose(1,2)
        x = self.dropout(x)

        return x

    def calc_same_padding(self, kernel_size):
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=in_channels)
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        x = F.pad(x, self.padding)
        x = self.conv(x)
        return self.conv_out(x)

class FeedForwardConformer(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)       
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, pos_emb, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        #q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # relative pos
        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1,2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1,2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1,2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2,-1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2,-1))
        matrix_bd = self.rel_shift(matrix_bd)

        matrix = matrix_ac+matrix_bd
        scores, attn = self.attention(matrix, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output, attn

    def rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def attention(self, matrix, v, d_k, mask=None, dropout=None):
    
        scores = matrix / math.sqrt(d_k)
    
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = scores.masked_fill(mask == 0, -2**15)
            attn = torch.softmax(attn, dim=-1) # (batch, head, time1, time2)
        else:
            attn = F.softmax(attn, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
    
        output = torch.matmul(attn, v)
        return output, attn

class RelativePositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=3000, xscale=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.xscale = xscale

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        
    def forward(self, x):
        x = x * self.xscale
        seq_len = x.shape[1]
        pe = self.pe[:,:seq_len].to(x.device)

        return self.dropout(x), self.dropout(pe)
