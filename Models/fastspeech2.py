#-*- coding: utf-8 -*-
from selectors import EpollSelector
from itsdangerous import NoneAlgorithm
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.postnets import PostConvNet
from Models.encoder import Encoder, ConformerEncoder
# FastSpeech 2 doesn't have decoder bacause of non-autoregressive model
# from Models.decoder import Decoder
from Models.varianceadaptor import VarianceAdaptor
# for debug
from Models.postnets import PostLowEnergyv1, PostLowEnergyv2
from Models.modules import SQEmbedding

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
                 d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, reduction_rate, dropout, dropout_postnet, dropout_variance_adaptor,
                 n_bins, f0_min, f0_max, energy_min, energy_max, pitch_pred=True, energy_pred=True, accent_emb=False, output_type=None, num_group=None, log_offset=1.,
                 multi_speaker=False, spk_emb_dim=None, spk_emb_architecture=None, debug=False):
        super().__init__()
        self.hp = hp
        self.spk_emb_architecture = spk_emb_architecture

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

        # new version
        self.use_sq_vae = hp.use_sq_vae
        if self.use_sq_vae:
            # variable for SQ-VAE
            self.param_var_q = "gaussian_1" # hp.param_var_q
            n_embeddings = 128 #64 # d_model_encoder
            embedding_dim = d_model_encoder #64
            self.embedding_dim = embedding_dim
            log_var_q_scalar = torch.Tensor(1)
            log_var_q_scalar.fill_(10.0).log_()
            self.register_parameter("log_var_q_scalar", nn.Parameter(log_var_q_scalar))
            self.codebook = SQEmbedding(self.param_var_q, n_embeddings, embedding_dim)
            # self.linear_sq = nn.Linear(embedding_dim, d_model_encoder)

        if self.hp.use_hop:
            self.hop_emb = nn.Embedding(3, d_model_encoder)

        if 'middle' in spk_emb_architecture:
            self.spk_proj = nn.Linear(spk_emb_dim, d_model_decoder)
        
        self.variance_adaptor = VarianceAdaptor(d_model_encoder, n_bins, f0_min, f0_max, energy_min, energy_max, log_offset, pitch_pred, energy_pred, dropout=dropout_variance_adaptor, use_rnn_length=self.hp.use_rnn_length, use_pos=self.hp.use_pos)

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

        if hp.postnet_pred:
            self.postnet = PostConvNet(hp=hp, num_hidden=d_model_decoder, mel_dim=trg_vocab, reduction_rate=reduction_rate,
                                       dropout=dropout_postnet)
        else:
            self.out = nn.Linear(d_model_decoder, trg_vocab * reduction_rate)
        #self.postnet = PostConvNet(d_model_decoder, trg_vocab, reduction_rate, 0.0, output_type, num_group)
        self.debug = debug
        if self.debug:
            self.post_model = PostLowEnergyv2(hp=hp, vocab_size=hp.mel_dim, out_size=hp.mel_dim_post, d_model=hp.d_model_encoder, N=hp.n_layer_post_model,
                                              #heads=hp.n_head_encoder, ff_conv_kernel_size=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.ff_conv_kernel_size_encoder, dropout=hp.dropout,
                                              heads=hp.n_head_encoder, ff_conv_kernel_size=hp.ff_conv_kernel_size_post, concat_after_encoder=hp.concat_after_post, dropout=dropout,
                                              multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim_postprocess, gender_emb=hp.gender_emb, speaker_emb=hp.speaker_emb, concat=hp.concat, spk_emb_postprocess_type=hp.spk_emb_postprocess_type,
                                              intermediate_layers_out=hp.intermediate_layers_out)
            if self.hp.version == 8 or self.hp.version == 9:
                self.post_model_replace_mask = PostLowEnergyv2(hp=hp, vocab_size=hp.mel_dim, out_size=hp.mel_dim_post, d_model=hp.d_model_encoder, N=hp.n_layer_post_model,
                                    heads=hp.n_head_encoder, ff_conv_kernel_size=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.concat_after_post, dropout=dropout,
                                    multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim_postprocess, gender_emb=hp.gender_emb, speaker_emb=hp.speaker_emb,
                                    concat=hp.concat, spk_emb_postprocess_type=hp.spk_emb_postprocess_type, intermediate_layers_out=hp.intermediate_layers_out)
            

    def forward(self, src, src_mask, mel_mask=None, d_target=None, p_target=None, e_target=None, accent=None, spkr_emb=None, fix_mask=None, spkr_emb_post=None,
                temperature=None, pitch_perturbation=False, duration_perturbation=False, hop_size=None):
        assert (self.training == True and pitch_perturbation == False) or (self.training == False)

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

        if 'middle' in self.spk_emb_architecture:
            spk_emb = self.spk_proj(F.normalize(spkr_emb)).unsqueeze(1)
            e_outputs = spk_emb + e_outputs

        if self.use_sq_vae:
            #TODO: dimension?
            # (B, T, H)
            assert (self.training is True and temperature is not None) or (self.training is False and temperature is None)
            z = e_outputs
            if self.param_var_q == "gaussian_1":
                log_var_q = self.log_var_q_scalar
            elif self.param_var_q == "gaussian_3" or self.param_var_q == "gaussian_4":
                log_var_q = z[:, :, self.embedding_dim:] + self.log_var_q_scalar
            else:
                raise Exception("Undefined param_var_q")

            z = z[:, :, :self.embedding_dim]

            if self.training:
                z, sq_vae_loss, sq_vae_perplexity, indices = self.codebook.encode(z, log_var_q)
            else:
                z, indices = self.codebook.encode(z, log_var_q)
                sq_vae_loss = None
                sq_vae_perplexity = None

            #z = z[0,0,:]
            e_outputs = z + e_outputs
            # import pdb; pdb.set_trace()
        else:
            sq_vae_loss = None
            sq_vae_perplexity = None

        if self.hp.use_hop:
            assert hop_size is not None
            # import pdb; pdb.set_trace()
            emb_hop = self.hop_emb(hop_size).unsqueeze(1)
            e_outputs = emb_hop + e_outputs

        if d_target is not None:
            variance_adaptor_output, log_d_prediction, p_prediction, e_prediction, _, _, text_dur_predicted = self.variance_adaptor(
                e_outputs, src_mask, mel_mask, d_target, p_target, e_target, p_scheduled_sampling=self.hp.p_scheduled_sampling)
        else:
            variance_adaptor_output, log_d_prediction, p_prediction, e_prediction, mel_len, mel_mask, text_dur_predicted = self.variance_adaptor(
                e_outputs, src_mask, mel_mask, d_target, p_target, e_target, p_scheduled_sampling=0.0, pitch_perturbation=pitch_perturbation, duration_perturbation=duration_perturbation)


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
            # 3: we assume that the model does not use spkr_emb_post
            #assert (self.training == True and spkr_emb_post is not None and self.hp.different_spk_emb_samespeaker == True) or (spkr_emb_post is not None and self.training == False) or (self.training == True and spkr_emb_post is None and self.hp.different_spk_emb_samespeaker == False)
            if self.hp.postnet_pred:
                input_meltomel = outputs_postnet
            else:
                input_meltomel = outputs_prenet

            if self.hp.semantic_mask and self.training:
                if self.hp.semantic_mask_phone:
                    mask_phone_feature = variance_adaptor_output
                else:
                    mask_phone_feature = None
                input_meltomel, mask_phone_feature, mask_frames = self._semantic_mask(input_meltomel, d_target, mask_phone_feature, p=self.hp.mask_probability)
                phone_feature = variance_adaptor_output if mask_phone_feature is None else mask_phone_feature
            else:
                phone_feature = variance_adaptor_output
                mask_frames = None


            if self.hp.version == 8:
                outputs_pro_post, ctc_outs, _ = self.post_model(outputs_prenet, mel_mask, variance_adaptor_output, spkr_emb=spkr_emb_post)
                outputs_pro_post_replace, ctc_outs, _ = self.post_model_replace_mask(input_meltomel, mel_mask, phone_feature, spkr_emb=spkr_emb_post)
                outputs_pro_post = (outputs_pro_post, outputs_pro_post_replace)
            elif self.hp.version == 9:
                outputs_pro_post, ctc_outs, _ = self.post_model(input_meltomel, mel_mask, phone_feature, spkr_emb=spkr_emb_post)
                outputs_pro_post_replace, ctc_outs, _ = self.post_model_replace_mask(input_meltomel, mel_mask, phone_feature, spkr_emb=spkr_emb_post)
                outputs_pro_post = (outputs_pro_post, outputs_pro_post_replace)
            elif self.hp.version == 10:
                ## TODO intemediate layers 
                outputs_pro_post, outputs_pro_post_replace, _ = self.post_model(input_meltomel, mel_mask, phone_feature, spkr_emb=spkr_emb_post)
                outputs_pro_post = (outputs_pro_post, outputs_pro_post_replace[0])
                ctc_outs = None
            else:
                outputs_pro_post, ctc_outs, _ = self.post_model(input_meltomel, mel_mask, phone_feature, spkr_emb=spkr_emb_post)
            
            # outputs, ctc_out, diff
            return outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, variance_adaptor_output, text_dur_predicted, attn_enc, attn_dec, outputs_pro_post, ctc_outs, mask_frames
        else:
            return outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, variance_adaptor_output, text_dur_predicted, attn_enc, attn_dec, None, None, None, sq_vae_loss, sq_vae_perplexity

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
