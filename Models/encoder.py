import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.layers import EncoderLayer, ConformerEncoderLayer
from Models.modules import PositionalEncoder, RelativePositionalEncoder
from Models.prenets import EncoderPreNet

class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args

def repeat(N, fn):
    return MultiSequential(*[fn() for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=True, accent_emb=False):
        super().__init__()
        self.N = N
        self.heads = heads
        self.accent_emb_flag = accent_emb

        if embedding:
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        else:
            self.embed = nn.Linear(vocab_size, d_model)

        if accent_emb:
            self.acc_embed = nn.Embedding(5, d_model, padding_idx=0)

        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = repeat(N, lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask, spkr_emb=None, accent=None):
        x = self.embed(src)
        if self.accent_emb_flag:
            accent_emb = self.acc_embed(accent)
            x = x + accent_emb

        x = self.pe(x)
        b, t, _ = x.shape
        attns = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn = self.layers[i](x, mask, spkr_emb)
            attns[:,i] = attn.detach()
        return self.norm(x), attns


class ConformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=True):
        super().__init__()
        if embedding:
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        else:
            self.embed = nn.Linear(vocab_size, d_model)
        self.N = N
        self.heads = heads
        xscale = 1
        self.pe = RelativePositionalEncoder(d_model, xscale=xscale, dropout=dropout)
        self.layers = repeat(self.N, lambda: ConformerEncoderLayer(d_model, self.heads, dropout, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask, spkr_emb=None):
        x = self.embed(src)
        x, pe = self.pe(x)
        b, t, _ = x.shape
        attns_enc = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn_enc = self.layers[i](x, pe, mask, spkr_emb)
            attns_enc[:,i] = attn_enc.detach()
        return self.norm(x), attns_enc
