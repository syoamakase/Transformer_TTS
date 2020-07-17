#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.layers import DecoderLayer
from Models.modules import PositionalEncoder
from Models.prenets import DecoderPreNet

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

class Decoder(nn.Module):
    # copy
    def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_decoder, dropout):
        super().__init__()
        self.N = N
        self.heads = heads
        # self.embed = nn.Linear(vocab_size, d_model)
        self.decoder_prenet = DecoderPreNet(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = repeat(N, lambda: DecoderLayer(d_model, heads, ff_conv_kernel_size, dropout, concat_after_decoder))
        self.norm = nn.LayerNorm(d_model)

    #@profile
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.decoder_prenet(trg)
        x = self.pe(x)
        b, t1, _ = x.shape
        b, t2, _ = e_outputs.shape
        attns_1 = torch.zeros((b, self.N, self.heads, t1, t1), device=x.device) # []
        attns_2 = torch.zeros((b, self.N, self.heads, t1, t2), device=x.device) # []
        #print('---------')
        for i in range(self.N):
            x, attn_1, attn_2 = self.layers[i](x, e_outputs, src_mask, trg_mask)
            attns_1[:,i] = attn_1.detach()
            attns_2[:,i] = attn_2.detach()
        return self.norm(x), attns_1, attns_2
