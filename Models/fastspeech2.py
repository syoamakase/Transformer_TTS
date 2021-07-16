#-*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Models.postnets import PostConvNet
from Models.encoder import Encoder
# FastSpeech 2 doesn't have decoder bacause of non-autoregressive model
# from Models.decoder import Decoder
from Models.varianceadaptor import VarianceAdaptor

class FastSpeech2(nn.Module):
    """ Model for FastSpeech 2
    Args:
        src_vocab (int): the dimension of input (text)
        trg_vocab (int): the dimension of output (mel)
        d_model_encoder (int): the dimension of hidden states of encoder
        N_e (int): the number of layers in the encoder
        n_head_encoder (int): the number of heads in each encoder layer
        ff_conv_kernel_size_encoder (int): kernel size of feed forward network in the encoder
        concat_after_encoder (bool): If True, concat the output of feed forward network and the input of encoder
        d_model_decoder (int): the dimension of hidden states of decoder
        N_d (int): the number of layers in the decoder
        n_head_decoder (int): the number of heads in each decoder layer
        ff_conv_kernel_size_decoder (int): kernel size of feed forward network in the ednoder
        concat_after_decoder (bool): If True, concat the output of feed forward network and the input of decoder
        reduction_rate (int): the number of frames which output at one decoder step
        dropout (float): dropout rate in the transformer
        CTC_training (bool): the multi task learning with CTC (it is ASR task)

    """
    def __init__(self, hp, src_vocab, trg_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder,
                d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, reduction_rate, dropout, dropout_postnet, CTC_training, 
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

        self.postnet = PostConvNet(hp=hp, num_hidden=d_model_decoder, mel_dim=trg_vocab, reduction_rate=reduction_rate, dropout=dropout_postnet, output_type=output_type, num_group=num_group)
        #self.postnet = PostConvNet(d_model_decoder, trg_vocab, reduction_rate, 0.0, output_type, num_group)

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
