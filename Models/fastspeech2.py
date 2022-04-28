#-*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Models.postnets import PostConvNet
from Models.encoder import Encoder, ConformerEncoder
# FastSpeech 2 doesn't have decoder bacause of non-autoregressive model
# from Models.decoder import Decoder
from Models.varianceadaptor import VarianceAdaptor
# for debug
from Models.postnets import PostLowEnergyv1, PostLowEnergyv2

import numpy as np

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

    """
    def __init__(self, hp, src_vocab, trg_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder,
                d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, reduction_rate, dropout, dropout_postnet, 
                n_bins, f0_min, f0_max, energy_min, energy_max, pitch_pred=True, energy_pred=True, accent_emb=False, output_type=None, num_group=None, log_offset=1.,
                multi_speaker=False, spk_emb_dim=None, spk_emb_architecture=None, debug=False):
        super().__init__()
        self.hp = hp

        if 'encoder' in spk_emb_architecture:
            if hp.encoder_type.lower() == 'conformer':
                self.encoder = ConformerEncoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder,
                                                dropout, multi_speaker, spk_emb_dim, True, accent_emb=accent_emb)
            else:
                self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder, 
                                       dropout, multi_speaker, spk_emb_dim, True, accent_emb)
        else:
            if hp.encoder_type.lower() == 'conformer':
                self.encoder = ConformerEncoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder,
                                                dropout, multi_speaker=False, spk_emb_dim=None, accent_emb=accent_emb)
            else:
                self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder,
                                       dropout, multi_speaker=False, spk_emb_dim=None, accent_emb=accent_emb)

        self.variance_adaptor = VarianceAdaptor(d_model_encoder, n_bins, f0_min, f0_max, energy_min, energy_max, log_offset, pitch_pred, energy_pred)

        if 'decoder' in spk_emb_architecture:
            if hp.decoder_type.lower() == 'conformer':
                self.decoder = ConformerEncoder(d_model_encoder, d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder,
                                                dropout, multi_speaker, spk_emb_dim, embedding=False, accent_emb=False)
            else:
                self.decoder = Encoder(d_model_encoder, d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder,
                                       dropout, multi_speaker, spk_emb_dim, embedding=False, accent_emb=False)
        else:
            if hp.decoder_type.lower() == 'conformer':
                self.decoder = ConformerEncoder(d_model_encoder, d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder,
                                                dropout, multi_speaker=False, spk_emb_dim=None, embedding=False, accent_emb=False)
            else:
                self.decoder = Encoder(d_model_encoder, d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder,
                                       dropout, multi_speaker=False, spk_emb_dim=None, embedding=False, accent_emb=False)

        # new version
        
        if hp.postnet_pred:
            self.postnet = PostConvNet(hp=hp, num_hidden=d_model_decoder, mel_dim=trg_vocab, reduction_rate=reduction_rate,
                                       dropout=dropout_postnet, output_type=output_type, num_group=num_group)
        else:
            self.out = nn.Linear(d_model_decoder, trg_vocab * reduction_rate)
        #self.postnet = PostConvNet(d_model_decoder, trg_vocab, reduction_rate, 0.0, output_type, num_group)
        self.debug = debug 
        if self.debug:
            self.post_model = PostLowEnergyv2(hp=hp, vocab_size=hp.mel_dim, out_size=hp.mel_dim_post, d_model=hp.d_model_encoder, N=hp.n_layer_post_model,
                                    heads=hp.n_head_encoder, ff_conv_kernel_size=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.ff_conv_kernel_size_encoder, dropout=hp.dropout,
                                    multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim_postprocess, gender_emb=hp.gender_emb, speaker_emb=hp.speaker_emb, concat=hp.concat,spk_emb_postprocess_type=hp.spk_emb_postprocess_type,
                                    layers_ctc_out=hp.layers_ctc_out)
            if self.hp.version == 8:
                self.post_model_replace_mask = PostLowEnergyv2(hp=hp, vocab_size=hp.mel_dim, out_size=hp.mel_dim_post, d_model=hp.d_model_encoder, N=hp.n_layer_post_model,
                                    heads=hp.n_head_encoder, ff_conv_kernel_size=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.ff_conv_kernel_size_encoder, dropout=hp.dropout,
                                    multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim_postprocess, gender_emb=hp.gender_emb, speaker_emb=hp.speaker_emb,
                                    concat=hp.concat,spk_emb_postprocess_type=hp.spk_emb_postprocess_type, layers_ctc_out=hp.layers_ctc_out)
            

    def forward(self, src, src_mask, mel_mask=None, d_target=None, p_target=None, e_target=None, accent=None, spkr_emb=None, fix_mask=None, spkr_emb_post=None):

        ### to make a mask that has fix len
        if fix_mask is not None:
            device = src.device
            size = src_mask.size(2)
            np_mask = np.zeros((1, size, size))#np.eye(size, k=0).astype('uint8')
            context_len = fix_mask #7
            for k in range(-(context_len-1)//2,(context_len-1)//2+1):
                np_mask[0] += np.eye(size, k=k)

            np_mask = torch.from_numpy(np_mask.astype('uint8') == 1).to(device)
            src_mask_fixlen = src_mask & np_mask
        else:
            src_mask_fixlen = src_mask

        e_outputs, attn_enc = self.encoder(src, src_mask_fixlen, spkr_emb, accent)

        if d_target is not None:
            variance_adaptor_output, log_d_prediction, p_prediction, e_prediction, _, _, text_dur_predicted = self.variance_adaptor(
                e_outputs, src_mask, mel_mask, d_target, p_target, e_target)
        else:
            variance_adaptor_output, log_d_prediction, p_prediction, e_prediction, mel_len, mel_mask, text_dur_predicted = self.variance_adaptor(
                e_outputs, src_mask, mel_mask, d_target, p_target, e_target)

        if fix_mask is not None:
            device = src.device
            size = mel_mask.size(2)
            np_mask = np.zeros((1, size, size))#np.eye(size, k=0).astype('uint8')
            context_len = fix_mask #7
            for k in range(-(context_len-1)//2,(context_len-1)//2+1):
                np_mask[0] += np.eye(size, k=k)

            np_mask = torch.from_numpy(np_mask.astype('uint8') == 1).to(device)
            mel_mask = mel_mask & np_mask

        d_output, attn_dec = self.decoder(variance_adaptor_output, mel_mask, spkr_emb, accent=None)
        if self.hp.postnet_pred:
            outputs_prenet, outputs_postnet = self.postnet(d_output)
        else:
            outputs_prenet = self.out(d_output)
            outputs_postnet = None

        if self.debug:
            assert (self.training == True and spkr_emb_post is not None and self.hp.different_spk_emb_samespeaker == True) or (spkr_emb_post is not None and self.training == False)
            if spkr_emb_post is None and (not self.hp.different_spk_emb_samespeaker):
                spkr_emb_post = spkr_emb
            if self.hp.postnet_pred:
                input_meltomel = outputs_postnet
            else:
                input_meltomel = outputs_prenet

            if self.hp.semantic_mask and self.training:
                if self.hp.semantic_mask_phone:
                    mask_phone_feature = variance_adaptor_output
                else:
                    mask_phone_feature = None
                input_meltomel, mask_phone_feature, mask_frames = self._semantic_mask(input_meltomel, d_target, mask_phone_feature)
                phone_feature = variance_adaptor_output if mask_phone_feature is None else mask_phone_feature
            else:
                phone_feature = variance_adaptor_output
                mask_frames = None

            outputs_pro_post, ctc_outs, _ = self.post_model(outputs_prenet, mel_mask, variance_adaptor_output, spkr_emb=spkr_emb_post)
            if self.hp.version == 8:
                outputs_pro_post_replace, ctc_outs, _ = self.post_model(input_meltomel, mel_mask, phone_feature, spkr_emb=spkr_emb_post)
                outputs_pro_post = (outputs_pro_post, outputs_pro_post_replace)
            # outputs, ctc_out, diff
            if self.training:
                return outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, variance_adaptor_output, text_dur_predicted, attn_enc, attn_dec, outputs_pro_post, ctc_outs, mask_frames
            else:
                return outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, variance_adaptor_output, text_dur_predicted, attn_enc, attn_dec, outputs_pro_post
        else:
            return outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, variance_adaptor_output, text_dur_predicted, attn_enc, attn_dec

    def _semantic_mask(self, mel, d_target, phone_feature=None, p=0.06, eps=1e-4):
        mask_frames = torch.zeros((mel.shape[0], mel.shape[1])).bool().to(mel.device).unsqueeze(-1)

        # (B, L)
        mask_sample = torch.rand(d_target.shape)
        d_target_cumsum = torch.cumsum(d_target, dim=1)
        # skip eos and sos
        for i in range(d_target_cumsum.shape[0]):
            for j in range(1, d_target_cumsum.shape[1]-1):
                if mask_sample[i,j] < p:
                    start = d_target_cumsum[i, j-1]
                    end = d_target_cumsum[i, j]
                    mask_frames[i, start:end] = True

        mel.masked_fill_(mask_frames, value=eps)
        if phone_feature is not None:
            phone_feature.masked_fill_(mask_frames, value=eps)
        return mel, phone_feature, mask_frames
