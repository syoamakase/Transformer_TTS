import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.layers import EncoderLayer, ConformerEncoderLayer, EncoderLayer_v2
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

def repeat_spkr_emb(N, fn1, fn2, layer):
    tmp = []
    for i in range(N):
        if i in layer:
            print('speaker_layer')
            tmp.append(fn2())
        else:   
            tmp.append(fn1())
    return MultiSequential(*tmp)

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=True, accent_emb=False, spk_emb_layer=None, gender_emb=False, layers_ctc_out=None):
        super().__init__()
        self.N = N
        self.heads = heads
        self.accent_emb_flag = accent_emb
        self.gender_emb_flag = gender_emb
        self.layers_ctc_out = layers_ctc_out

        if embedding:
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        else:
            self.embed = nn.Linear(vocab_size, d_model)

        if accent_emb:
            self.acc_embed = nn.Embedding(12, d_model)

        if gender_emb:
            self.gender_embed = nn.Embedding(2, d_model)

        self.pe = PositionalEncoder(d_model, dropout=dropout)
        if spk_emb_layer is None:
            self.layers = repeat(N, lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim))
        else:
            if version == 1:
                self.layers = repeat_spkr_emb(N, lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder),
                lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim), spk_emb_layer)
            else:
                self.layers = repeat_spkr_emb(N, lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder),
                lambda: EncoderLayer_v2(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim), spk_emb_layer)
        self.norm = nn.LayerNorm(d_model)

        if self.layers_ctc_out:
            self.ctc_out_layers = nn.ModuleList([nn.Linear(d_model, 152) for n in range(len(self.layers_ctc_out))])

    def forward(self, src, mask, spkr_emb=None, accent=None, gender_id=None):
        x = self.embed(src)
        if self.accent_emb_flag:
            accent_emb = self.acc_embed(accent)
            x = x + accent_emb

        if self.gender_emb_flag:
            assert gender_id is not None, "gender_id is None"
            gender_emb = self.gender_emb(gender_id)
            x = x + gender_emb

        x = self.pe(x)
        ctc_outs = []
        b, t, _ = x.shape
        attns = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn = self.layers[i](x, mask, spkr_emb)
            if self.layers_ctc_out:
                if i in self.layers_ctc_out:
                    i_ctc = len(ctc_outs)
                    ctc_out = self.ctc_out_layers[i_ctc](x)
                    ctc_outs.append(ctc_out)
            attns[:,i] = attn.detach()
        if self.layers_ctc_out:
            return self.norm(x), attns, ctc_outs
        else:
            return self.norm(x), attns


class ConformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=True, accent_emb=False):
        super().__init__()
        self.accent_emb_flag = accent_emb
        if accent_emb:
            self.acc_embed = nn.Embedding(12, d_model, padding_idx=0)
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

    def forward(self, src, mask, spkr_emb=None, accent=None):
        x = self.embed(src)
        if self.accent_emb_flag:
            accent_emb = self.acc_embed(accent)
            x = x + accent_emb

        x, pe = self.pe(x)
        b, t, _ = x.shape
        attns_enc = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn_enc = self.layers[i](x, pe, mask, spkr_emb)
            attns_enc[:,i] = attn_enc.detach()
        return self.norm(x), attns_enc

class EncoderPostprocessing(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=True, accent_emb=False, gender_emb=False, speaker_emb=False, ctc_out=False):
        super().__init__()
        self.N = N
        self.heads = heads
        self.accent_emb_flag = accent_emb
        self.gender_emb_flag = gender_emb
        self.speaker_emb_flag = speaker_emb
        self.ctc_flag = ctc_out

        if embedding:
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        else:
            self.embed = nn.Linear(vocab_size, d_model)

        if accent_emb:
            self.acc_embed = nn.Embedding(5, d_model, padding_idx=0)

        if gender_emb:
            self.gender_embed = nn.Embedding(2, d_model)

        if speaker_emb:
            self.speaker_embed = nn.Embedding(247, d_model)

        if ctc_out:
            self.ctc_linear = nn.Linear(d_model, 152)

        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = repeat(N, lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask, spkr_emb=None, accent=None, gender=None):
        x = self.embed(src)
        if self.accent_emb_flag:
            accent_emb = self.acc_embed(accent)
            x = x + accent_emb

        if self.gender_emb_flag:
            assert gender is not None, "gender is None"
            gender_emb = self.gender_embed(gender)
            #v1
            print('gender')
            x = x + gender_emb
            # v2
            #x = x + F.softsign(gender_emb)

        if self.speaker_emb_flag:
            speaker_emb = self.speaker_embed(spkr_emb)
            x = x + speaker_emb.unsqueeze(1)

        x = self.pe(x)
        b, t, _ = x.shape
        attns = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn = self.layers[i](x, mask, spkr_emb)
            if i == 2 and self.ctc_flag:
                ctc_out = self.ctc_linear(x)
            else:
                ctc_out = None
            attns[:,i] = attn.detach()
        return self.norm(x), ctc_out, attns
