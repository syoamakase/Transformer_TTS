import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.layers import EncoderLayer
from Models.modules import PositionalEncoder
from Models.prenets import EncoderPreNet

class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args

def repeat(N, fn):
    """Repeat module N times.
    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn() for _ in range(N)])

class Encoder(nn.Module):
    # copy
    def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout):
        super().__init__()
        self.N = N
        self.heads = heads
        # self.embed = EncoderPreNet(vocab_size, d_model)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = repeat(N, lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout, concat_after_encoder)) 
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        #print('encoder outputs', x.max(), x.min(), x)
        x = self.pe(x)
        b, t, _ = x.shape
        attns = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            attns[:,i] = attn.detach()
        return self.norm(x), attns
