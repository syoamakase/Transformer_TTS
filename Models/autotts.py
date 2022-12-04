#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.aligner import Aligner
from Models.encoder import Encoder

class AutoTTS(nn.Module):

    def __init__(self, hp, src_vocab, trg_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, 
                concat_after_encoder, d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder,
                concat_after_decoder, reduction_rate, dropout, dropout_postnet, n_bins, f0_min, f0_max,
                energy_min, energy_max, pitch_pred=True, energy_pred=True, accent_emb=False, output_type=None,
                num_group=None, log_offset=1.,
                multi_speaker=False, spk_emb_dim=None, spk_emb_architecture=None, debug=False, M=50):
        super.__init__()
        
        self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder,
                               concat_after_encoder, dropout, multi_speaker, spk_emb_dim, True, accent_emb)
        self.aligner = Aligner(hp, M)
    
    def forward(self, src, src_mask, spkr_emb=None, accent=None):

        e_outputs, attn_enc = self.encoder(src, src_mask, spkr_emb, accent)
        
        p_duration_phone = self.aligner(e_outputs)

    
    def _get_attention_weights(self, p_duration_phone):
        """
        Args:
            p_duration_phone (torch.tensor): phone probabilties by sigmoid (B x L x M).
        """
        b, l, _ = p_duration_phone.shape
        p_cum_prod = torch.cat(torch.ones((b,l,1)).to(p_duration_phone.device), p_duration_phone, dim=2)
        p_lengths = p_duration_phone * p_cum_prod[:,:-1,:]

        
