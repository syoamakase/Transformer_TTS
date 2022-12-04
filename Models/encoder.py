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
            print('kernel 5')
            tmp.append(fn2())
        else:
            tmp.append(fn1())
    return MultiSequential(*tmp)

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 N,
                 heads,
                 ff_conv_kernel_size,
                 concat_after_encoder,
                 dropout,
                 multi_speaker=False,
                 spk_emb_dim=None,
                 embedding=True,
                 accent_emb=False, 
                 spk_emb_layer=None,
                 gender_emb=False,
                 intermediate_layers_out=None):
        super().__init__()
        self.N = N
        self.heads = heads
        self.accent_emb_flag = accent_emb
        self.gender_emb_flag = gender_emb
        self.intermediate_layers_out = intermediate_layers_out

        if embedding:
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        else:
            self.embed = nn.Linear(vocab_size, d_model)

        if accent_emb:
            #self.acc_embed = nn.Embedding(10, d_model)
            self.acc_embed = nn.Embedding(5, d_model)

        if gender_emb:
            self.gender_embed = nn.Embedding(2, d_model)

        self.pe = PositionalEncoder(d_model, dropout=dropout)
        if spk_emb_layer is None:
            self.layers = repeat(N, lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim))
            # self.layers = repeat_spkr_emb(N, lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim),
            #                              lambda: EncoderLayer(d_model, heads, 5, dropout=dropout, concat_after=concat_after_encoder, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim), layer=[6,7,8,9,10,11])
        #else:
        #    if version == 1:
        #        self.layers = repeat_spkr_emb(N, lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder),
        #        lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim), spk_emb_layer)
        #    else:
        #        self.layers = repeat_spkr_emb(N, lambda: EncoderLayer(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder),
        #        lambda: EncoderLayer_v2(d_model, heads, ff_conv_kernel_size, dropout=dropout, concat_after=concat_after_encoder, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim), spk_emb_layer)
        self.norm = nn.LayerNorm(d_model)

        if self.intermediate_layers_out:
            self.intermediate_layers = nn.ModuleList([nn.Linear(d_model, 80) for n in range(len(self.intermediate_layers_out))])

    def forward(self, src, mask, spkr_emb=None, accent=None, gender_id=None, attn_detach=True):
        x = self.embed(src)
        # if self.accent_emb_flag:
        #     accent_emb = self.acc_embed(accent)
        #     x = x + accent_emb

        # if self.gender_emb_flag:
        #     assert gender_id is not None, "gender_id is None"
        #     gender_emb = self.gender_emb(gender_id)
        #     x = x + gender_emb

        x = self.pe(x)
        intermediate_outs = []
        b, t, _ = x.shape
        attns = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn = self.layers[i](x, mask, spkr_emb)
            if self.intermediate_layers_out is not None:
                if i in self.intermediate_layers_out:
                    i_intermediate = len(intermediate_outs)
                    intermediate_out = self.intermediate_layers[i_intermediate](x)
                    intermediate_outs.append(intermediate_out)
            attns[:, i] = attn if attn_detach else attn.detach()
        if self.accent_emb_flag:
            accent_emb = self.acc_embed(accent)
            x = x + accent_emb
        if len(intermediate_outs) > 0:
            return self.norm(x), attns, intermediate_outs
        else:
            return self.norm(x), attns


class ConformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=True, accent_emb=False, spk_emb_layer=None, gender_emb=False, intermediate_layers_out=None):
        """ATTENTION!: the architecture is slightly different from ASR one.

        Args:
            vocab_size (_type_): _description_
            d_model (_type_): _description_
            N (_type_): _description_
            heads (_type_): _description_
            dropout (_type_): _description_
            multi_speaker (bool, optional): _description_. Defaults to False.
            spk_emb_dim (_type_, optional): _description_. Defaults to None.
            embedding (bool, optional): _description_. Defaults to True.
            accent_emb (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.accent_emb_flag = accent_emb
        if embedding:
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        else:
            self.embed = nn.Linear(vocab_size, d_model)

        if accent_emb:
            self.acc_embed = nn.Embedding(13, d_model)

        self.N = N
        self.heads = heads
        xscale = 1
        self.pe = RelativePositionalEncoder(d_model, xscale=xscale, dropout=dropout)
        self.layers = repeat(self.N, lambda: ConformerEncoderLayer(d_model, self.heads, ff_conv_kernel_size=ff_conv_kernel_size, dropout=dropout,  multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask, spkr_emb=None, accent=None, attn_detach=True):
        x = self.embed(src)
        if self.accent_emb_flag:
            accent_emb = self.acc_embed(accent)

            x = x + accent_emb

        x, pe = self.pe(x)
        b, t, _ = x.shape
        attns_enc = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn_enc = self.layers[i](x, pe, mask, spkr_emb)
            attns_enc[:,i] = attn_enc if attn_detach else attn_enc.detach()
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

    def forward(self, src, mask, spkr_emb=None, accent=None, gender=None, attn_detach=True):
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
            attns[:,i] = attn if attn_detach else attn.detach()
        return self.norm(x), ctc_out, attns
