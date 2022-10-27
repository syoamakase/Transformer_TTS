from operator import truediv
from pickle import FALSE
from selectors import EpollSelector
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.postnets import PostConvNet
from Models.prenets import DecoderPreNet
from Models.encoder import Encoder, ConformerEncoder
from Models.decoder import Decoder, Tacotron2Decoder

from Models.gst import StyleEmbedding

class Transformer(nn.Module):
    # def __init__(self, src_vocab, trg_vocab, d_model, N_e, N_d, heads, dropout):
    def __init__(self,
                 hp,
                 src_vocab,
                 trg_vocab,
                 d_model_encoder,
                 N_e,
                 n_head_encoder,
                 ff_conv_kernel_size_encoder,
                 concat_after_encoder,
                 d_model_decoder,
                 N_d,
                 n_head_decoder,
                 ff_conv_kernel_size_decoder,
                 concat_after_decoder,
                 reduction_rate,
                 dropout,
                 dropout_prenet=0.5,
                 dropout_postnet=0.5,
                 multi_speaker=False,
                 spk_emb_dim=None,
                 spk_emb_architecture=None,
                 output_type=None,
                 decoder_type='transformer'
):
        super().__init__()

        self.gst = hp.gst
        self.spk_emb_vers = 1
        self.decoder_type = decoder_type

        if 'encoder' in spk_emb_architecture and self.spk_emb_vers == 1:
            multi_speaker_encoder = True
        else:
            multi_speaker_encoder = False
        # else:
        if hp.encoder_type.lower() == 'conformer':
            #self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None)
            self.encoder = ConformerEncoder(src_vocab, d_model_encoder, N_e, n_head_encoder,
                                            ff_conv_kernel_size_encoder, concat_after_encoder, dropout,
                                            multi_speaker=multi_speaker_encoder, spk_emb_dim=spk_emb_dim)
        else:
            self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder,
                                   ff_conv_kernel_size_encoder, concat_after_encoder, dropout=dropout,
                                   multi_speaker=multi_speaker_encoder, spk_emb_dim=spk_emb_dim)

        if d_model_encoder != d_model_decoder:
            self.linear = nn.Linear(d_model_encoder, d_model_decoder)
        else:
            self.linear = None

        if self.gst:
            self.style_embedding = StyleEmbedding(hp)

        if multi_speaker and self.spk_emb_vers == 2:
            self.spk_proj = nn.Linear(spk_emb_dim, d_model_decoder)

        if 'decoder' in spk_emb_architecture and self.spk_emb_vers == 1:
            multi_speaker_decoder = True
        else:
            multi_speaker_decoder = False
        if self.decoder_type.lower() == 'transformer':
            self.decoder = Decoder(trg_vocab, d_model_decoder, N_d, n_head_decoder,
                                   ff_conv_kernel_size_decoder, concat_after_decoder, dropout=dropout,
                                   dropout_prenet=dropout_prenet, multi_speaker=multi_speaker_decoder,
                                   spk_emb_dim=spk_emb_dim, output_type=output_type)

            self.out = nn.Linear(d_model_decoder, trg_vocab*reduction_rate)
            self.stop_token = nn.Linear(d_model_decoder, reduction_rate)
        else:
            self.decoder = Tacotron2Decoder(trg_vocab, d_model_decoder, d_model_encoder, reduction_rate, dropout_prenet=dropout_prenet,
                                            multi_speaker=multi_speaker_decoder, spk_emb_dim=spk_emb_dim, output_type=output_type, zoneout_rate=0.1)
        self.postnet = PostConvNet(hp, d_model_decoder, trg_vocab, reduction_rate, dropout_postnet, prev_version=False)

    def forward(self, src, trg, src_mask, trg_mask, spkr_emb, training=True, ref_mel=None):

        e_outputs, attn_enc = self.encoder(src, src_mask, spkr_emb=spkr_emb, attn_detach=False)
        if self.linear is not None:
            e_outputs = self.linear(e_outputs)

        if self.gst:
            if training:
                spk_embed = self.style_embedding(trg, trg_mask)
            else:
                spk_embed = self.style_embedding(ref_mel, None)
            e_outputs = e_outputs + spk_embed

        if self.spk_emb_vers == 2:
            spk_emb = self.spk_proj(F.normalize(spkr_emb)).unsqueeze(1)
            e_outputs = spk_emb + e_outputs

        if self.decoder_type.lower() == 'transformer':
            d_output, attn_dec_dec, attn_dec_enc = self.decoder(trg, e_outputs, src_mask, trg_mask, spkr_emb, attn_detach=False)

            outputs_prenet = self.out(d_output)
            outputs_postnet = self.postnet(outputs_prenet)
            stop_token = self.stop_token(d_output).squeeze(2)
        else:
            outputs_prenet, stop_token, attn_dec_enc = self.decoder(trg, e_outputs, speaker_emb=spkr_emb)
            outputs_postnet = self.postnet(outputs_prenet)
            attn_dec_dec = None

        return outputs_prenet, outputs_postnet, stop_token, attn_enc, attn_dec_dec, attn_dec_enc
