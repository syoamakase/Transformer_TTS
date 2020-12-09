#-*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Models.postnets import PostConvNet
from Models.encoder import Encoder
# FastSpeech 2 doesn't have decoder bacause of non-autoregressive model
# from Models.decoder import Decoder
from Models.varianceadaptor import VarianceAdaptor

class FastSpeech2(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder,
                d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, reduction_rate, dropout, CTC_training, 
                n_bins, f0_min, f0_max, energy_min, energy_max, pitch_pred=True, energy_pred=True, output_type=None, num_group=None, log_offset=1.,
                multi_speaker=False, spk_emb_dim=None, spkr_emb=None):
        super().__init__()

        if 'encoder' in spkr_emb:
            self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder, dropout, multi_speaker, spk_emb_dim)
        else:
            self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None)

        self.variance_adaptor = VarianceAdaptor(d_model_encoder, n_bins, f0_min, f0_max, energy_min, energy_max, log_offset, pitch_pred, energy_pred)

        if 'decoder' in spkr_emb:
            self.decoder = Encoder(d_model_encoder, d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, dropout, multi_speaker, spk_emb_dim, embedding=False)
        else:
            self.decoder = Encoder(d_model_encoder, d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=False)

        self.postnet = PostConvNet(d_model_decoder, trg_vocab, reduction_rate, 0.5, output_type, num_group)

    def forward(self, src, src_mask, mel_mask=None, d_target=None, p_target=None, e_target=None, spkr_emb=None):
        e_outputs, attn_enc = self.encoder(src, src_mask, spkr_emb)

        if d_target is not None:
            variance_adaptor_output, log_d_prediction, p_prediction, e_prediction, _, _ = self.variance_adaptor(
                e_outputs, src_mask, mel_mask, d_target, p_target, e_target)
        else:
            variance_adaptor_output, log_d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                e_outputs, src_mask, mel_mask, d_target, p_target, e_target)

        d_output, attn_dec = self.decoder(variance_adaptor_output, mel_mask, spkr_emb)
        outputs_prenet, outputs_postnet = self.postnet(d_output)

        return outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, attn_enc, attn_dec
