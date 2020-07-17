#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.modules import MultiHeadAttention, FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, ff_conv_kernel_size, dropout=0.1, concat_after=False):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        # self.norm_3 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout, concat_after=concat_after)
        self.ff = FeedForward(d_model, ff_conv_kernel_size, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # res = x
        # # x = self.norm_1(x)
        # out, attn = self.attn(x,x,x,mask, True)
        # x = res + self.dropout_1(out)
        # x = self.norm_2(x)
        # res = x
        # x = res + self.dropout_2(self.ff(x))
        # return self.norm_3(x), attn

        ## not good ???
        res = x
        x = self.norm_1(x)
        out, attn = self.attn(x,x,x,mask, True)
        x = res + self.dropout_1(out)
        res = x
        x = self.norm_2(x)
        x = res + self.dropout_2(self.ff(x))
        return x, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, ff_conv_kernel_size, dropout=0.1, concat_after=False):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout, concat_after=concat_after)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout, concat_after=concat_after)
        self.ff = FeedForward(d_model, ff_conv_kernel_size, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        # res = x 
        # out, attn_1 = self.attn_1(x, x, x, trg_mask)
        # x = res + self.dropout_1(out)
        # x = self.norm_1(x)

        # res = x
        # out, attn_2 = self.attn_2(x, e_outputs, e_outputs, src_mask)
        # x = res + self.dropout_2(out)
        # x = self.norm_2(x)

        # res = x
        # x = res + self.dropout_3(self.ff(x))
        # return self.norm_3(x), attn_1, attn_2
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

        x = res + self.dropout_3(self.ff(x))
        return x, attn_1, attn_2