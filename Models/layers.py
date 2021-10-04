#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.modules import MultiHeadAttention, FeedForward, FeedForwardConformer, ConvolutionModule, RelativeMultiHeadAttention 

class EncoderLayer(nn.Module):
    #def __init__(self, d_model, heads, ff_conv_kernel_size, dropout=0.1, concat_after=False):
    def __init__(self, d_model, heads, ff_conv_kernel_size, dropout=0.1, concat_after=False, multi_speaker=False, spk_emb_dim=None):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        # self.norm_3 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, d_model, d_model, d_model, dropout=dropout, concat_after=concat_after)
        self.ff = FeedForward(d_model, ff_conv_kernel_size, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.multi_speaker = multi_speaker
        if self.multi_speaker:
            self.multi_emb = nn.Embedding(spk_emb_dim, d_model)
            self.speaker_L_l1_es = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask, spkr_emb=None):
        res = x
        x = self.norm_1(x)
        out, attn = self.attn(x,x,x,mask, False)
        x = res + self.dropout_1(out)
        res = x
        x = self.norm_2(x)
        if self.multi_speaker:
            #print('enc')
            spkr_embeds_dec = self.multi_emb(spkr_emb)
            x = x + F.softsign(self.speaker_L_l1_es(spkr_embeds_dec)).unsqueeze(1)
        x = res + self.dropout_2(self.ff(x))
        return x, attn


class ConformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, ff_conv_kernel_size, dropout=0.1, multi_speaker=False, spk_emb_dim=None):
        ## TODO: add ff_conv_kernel_size
        super().__init__()
        self.ff_1 = FeedForwardConformer(d_model, d_ff=d_model*4, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.attn = RelativeMultiHeadAttention(heads, d_model, dropout=dropout)
        self.conv_module = ConvolutionModule(d_model, dropout=dropout)
        self.ff_2 = FeedForwardConformer(d_model, d_ff=d_model*4, dropout=dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.multi_speaker = multi_speaker
        if self.multi_speaker:
            self.multi_emb = nn.Embedding(spk_emb_dim, d_model)
            self.speaker_L_l1_es = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, pe, mask, spkr_emb=None):
        ## TODO: spkr emb
        x = x + 0.5 * self.ff_1(x)
        res = x
        x = self.norm(x)
        x, attn_enc_enc = self.attn(x,x,x,pe,mask)
        x = res + self.dropout_1(x)
        x = x + self.conv_module(x)

        if self.multi_speaker:
            print('enc conf')
            spkr_embeds_dec = self.multi_emb(spkr_emb)
            x = x + F.softsign(self.speaker_L_l1_es(spkr_embeds_dec)).unsqueeze(1)
        x = x + self.dropout_2(self.ff_2(x))
        return x, attn_enc_enc


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, ff_conv_kernel_size, dropout=0.1, concat_after=False, multi_speaker=False, spk_emb_dim=None):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        #self.norm_4 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, d_model, d_model, d_model, dropout=dropout, concat_after=concat_after)
        self.attn_2 = MultiHeadAttention(heads, d_model, d_model, d_model, d_model, dropout=dropout, concat_after=concat_after)
        self.ff = FeedForward(d_model, ff_conv_kernel_size, dropout=dropout)

        self.multi_speaker = multi_speaker
        if self.multi_speaker:
            self.multi_emb = nn.Embedding(spk_emb_dim, d_model)
            self.speaker_L_l1_es = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, e_outputs, src_mask, trg_mask, spkr_emb=None):
        res = x
        x = self.norm_1(x)
        out, attn_1 = self.attn_1(x, x, x, trg_mask, True)
        x = res + self.dropout_1(out)
        res = x
        x = self.norm_2(x)
        out, attn_2 = self.attn_2(x, e_outputs, e_outputs, src_mask, True)
        x = res + self.dropout_2(out)
        res = x
        x = self.norm_3(x)
        if self.multi_speaker:
            spkr_embeds_dec = self.multi_emb(spkr_emb)
            x = x + F.softsign(self.speaker_L_l1_es(spkr_embeds_dec)).unsqueeze(1)
        x = res + self.dropout_3(self.ff(x))
        return x, attn_1, attn_2
        #return self.norm_4(x), attn_1, attn_2
