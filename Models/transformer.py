import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.postnets import PostConvNet
from Models.encoder import Encoder, ConformerEncoder
from Models.decoder import Decoder

from Models.gst import StyleEmbedding

class Transformer(nn.Module):
    # def __init__(self, src_vocab, trg_vocab, d_model, N_e, N_d, heads, dropout):
    def __init__(self, hp, src_vocab, trg_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder,
                d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, reduction_rate, dropout, dropout_prenet=0.5, dropout_postnet=0.5,
                CTC_training=False, multi_speaker=False, spk_emb_dim=None, output_type=None, num_group=None, spkr_emb=None):
        super().__init__()

        self.CTC_training = CTC_training
        self.gst = hp.gst

        if 'encoder' in spkr_emb:
            self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder, dropout, multi_speaker, spk_emb_dim)
            #self.encoder = ConformerEncoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder, dropout, multi_speaker, spk_emb_dim)
        else:
            if hp.encoder_type == 'conformer':
                #self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None)
                self.encoder = ConformerEncoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None)
            else:
                self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None)

        if d_model_encoder != d_model_decoder:
            self.linear = nn.Linear(d_model_encoder, d_model_decoder)
        else:
            self.linear = None
        
        if self.gst:
            self.style_embedding = StyleEmbedding(hp)

        if 'decoder' in spkr_emb:
            self.decoder = Decoder(trg_vocab, d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, dropout, dropout_prenet, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim, output_type=output_type)
        else:
            self.decoder = Decoder(trg_vocab, d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, dropout, dropout_prenet, multi_speaker=multi_speaker, spk_emb_dim=spk_emb_dim, output_type=output_type)

        self.stop_token = nn.Linear(d_model_decoder, reduction_rate)
        if self.CTC_training:
            self.CTC_linear = nn.Linear(d_model_decoder, src_vocab)
        self.postnet = PostConvNet(hp, d_model_decoder, trg_vocab, reduction_rate, dropout_postnet, output_type=output_type, num_group=num_group)

    def forward(self, src, trg, src_mask, trg_mask, spkr_emb, training=True, ref_mel=None):
        e_outputs, attn_enc = self.encoder(src, src_mask)
        if self.linear is not None:
            e_outputs = self.linear(e_outputs)

        if self.gst:
            if training:
                spk_embed = self.style_embedding(trg, trg_mask)
            else:
                spk_embed = self.style_embedding(ref_mel, None)
            e_outputs = e_outputs + spk_embed

        d_output, attn_dec_dec, attn_dec_enc = self.decoder(trg, e_outputs, src_mask, trg_mask, spkr_emb)

        outputs_prenet, outputs_postnet = self.postnet(d_output)
        stop_token = self.stop_token(d_output).squeeze(2)
        if self.CTC_training:
            ctc_outputs = self.CTC_linear(d_output)
            if training:
                return outputs_prenet, outputs_postnet, stop_token, attn_enc, attn_dec_dec, attn_dec_enc, ctc_outputs
            else:
                return outputs_prenet, outputs_postnet, stop_token, attn_enc, attn_dec_dec, attn_dec_enc
        else:
            return outputs_prenet, outputs_postnet, stop_token, attn_enc, attn_dec_dec, attn_dec_enc
