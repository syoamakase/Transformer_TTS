import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.postnets import PostConvNet
from Models.encoder import Encoder
from Models.decoder import Decoder

class Transformer(nn.Module):
    # def __init__(self, src_vocab, trg_vocab, d_model, N_e, N_d, heads, dropout):
    def __init__(self, src_vocab, trg_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder,
                d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, reduction_rate, dropout, CTC_training=False):
        super().__init__()
        self.CTC_training = CTC_training
        # self.cnn_encoder = CNN_embedding(src_vocab, hp.cnn_dim)
        self.encoder = Encoder(src_vocab, d_model_encoder, N_e, n_head_encoder, ff_conv_kernel_size_encoder, concat_after_encoder, dropout)
        self.decoder = Decoder(trg_vocab, d_model_decoder, N_d, n_head_decoder, ff_conv_kernel_size_decoder, concat_after_decoder, dropout)
        self.stop_token = nn.Linear(d_model_decoder, reduction_rate)
        if self.CTC_training:
            self.CTC_linear =  nn.Linear(d_model_decoder, src_vocab)
        self.postnet = PostConvNet(d_model_decoder, trg_vocab, reduction_rate)

    def forward(self, src, trg, src_mask, trg_mask):

        e_outputs, attn_enc = self.encoder(src, src_mask)
        d_output, attn_dec_dec, attn_dec_enc = self.decoder(trg, e_outputs, src_mask, trg_mask)
        outputs_prenet, outputs_postnet = self.postnet(d_output)
        stop_token = self.stop_token(d_output).squeeze(2)
        if self.CTC_training:
            ctc_outputs = self.CTC_linear(d_output)
            return outputs_prenet, outputs_postnet, stop_token, attn_enc, attn_dec_dec, attn_dec_enc, ctc_outputs
        else:
            return outputs_prenet, outputs_postnet, stop_token, attn_enc, attn_dec_dec, attn_dec_enc