#-*- coding: utf-8 -*-
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

#from Models.varianoce_predictor import VarianceAdaptor
from Models.encoder import Encoder, EncoderPostprocessing, ConformerEncoder

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PostConvNet(nn.Module):
    """
    Post Convolutional Network (mel --> mel)
    """
    def __init__(self, hp, num_hidden, mel_dim, reduction_rate, dropout=0.5, output_type=None, num_group=None):
        super(PostConvNet, self).__init__()
        self.output_type = output_type
        if output_type:
            self.conv1 = nn.Conv1d(in_channels=mel_dim*reduction_rate*num_group,
                                   out_channels=num_hidden,
                                   kernel_size=5,
                                   padding=4)
            self.conv_list = clones(nn.Conv1d(in_channels=num_hidden,
                                            out_channels=num_hidden,
                                            kernel_size=5,
                                            padding=4), 3)
            self.conv2 = nn.Conv1d(in_channels=num_hidden,
                                out_channels=mel_dim*reduction_rate*num_group,
                                kernel_size=5,
                                padding=4)

            self.out1 = nn.Linear(num_hidden, mel_dim*reduction_rate)
            self.out2 = nn.Linear(num_hidden, mel_dim*reduction_rate)

        else:
            print(mel_dim, reduction_rate)
            self.conv1 = nn.Conv1d(in_channels=mel_dim*reduction_rate,
                                  out_channels=num_hidden,
                                  kernel_size=5,
                                  padding=4)
            self.conv_list = clones(nn.Conv1d(in_channels=num_hidden,
                                            out_channels=num_hidden,
                                            kernel_size=5,
                                            padding=4), 3)
            self.conv2 = nn.Conv1d(in_channels=num_hidden,
                                   out_channels=mel_dim * reduction_rate,
                                   kernel_size=5,
                                   padding=4)

            self.out = nn.Linear(num_hidden, mel_dim * reduction_rate)

        ## dev
        #self.pitch_pred = hp.pitch_pred
        #self.energy_pred = hp.energy_pred
        #if self.pitch_pred:
        #    n_bins = 256
        #    self.pitch_predictor = VariancePredictor(d_model_encoder)
        #    self.pitch_bins = torch.exp(torch.linspace(np.log(f0_min), np.log(f0_max), n_bins-1
        #    self.pitch_embedding = nn.Embedding(n_bins, hp.d_model_d)
        #if self.energy_pred:
        #    n_bins = 256
        #    self.energy_predictor = VariancePredictor(d_model_encoder)
        #    self.energy_bins = torch.linspace(energy_min, energy_max, n_bins-1).to(DEVICE)
        #    self.energy_embedding = nn.Embedding(n_bins, d_model_encoder)

        self.batch_norm_list = clones(nn.BatchNorm1d(num_hidden), 3)
        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(3)])

    def forward(self, input_, mask=None):
        if self.output_type:
            mel_pred1 = self.out1(input_).transpose(1, 2)
            mel_pred2 = self.out2(input_).transpose(1, 2)
            # Causal Convolution (for auto-regressive)
            mel_pred = torch.cat((mel_pred1, mel_pred2), dim=1)
            input_ = self.dropout1(torch.tanh(self.pre_batchnorm(self.conv1(mel_pred)[:, :, :-4])))
            for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
                input_ = dropout(torch.tanh(batch_norm(conv(input_)[:, :, :-4])))
            input_ = self.conv2(input_)[:, :, :-4]
            input_ = mel_pred + input_
            return mel_pred.transpose(1, 2), input_.transpose(1, 2)
        else:
            ## older version (until 2022/1/13)
            mel_pred = self.out(input_).transpose(1, 2)
            # Causal Convolution (for auto-regressive)
            input_ = self.dropout1(torch.tanh(self.pre_batchnorm(self.conv1(mel_pred)[:, :, :-4])))
            for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
                input_ = dropout(torch.tanh(batch_norm(conv(input_)[:, :, :-4])))
            input_ = self.conv2(input_)[:, :, :-4]
            input_ = mel_pred + input_
            return mel_pred.transpose(1, 2), input_.transpose(1, 2)


class PostLowEnergyv1(nn.Module):
    def __init__(self, vocab_size, out_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=False):
        super(PostLowEnergyv1, self).__init__()
        ##def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=True):

        self.encoder = Encoder(vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, embedding=embedding)

        self.out = nn.Linear(d_model, out_size)

    def forward(self, src, src_mask, spkr_emb=None):
        e_outputs, attn_enc = self.encoder(src, src_mask, spkr_emb)

        outputs = self.out(e_outputs)
        
        return outputs

class PostLowEnergyv2(nn.Module):
    def __init__(self, hp, vocab_size, out_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=False, spk_emb_layer=None, gender_emb=False, speaker_emb=False, concat=False, spk_emb_postprocess_type=None, layers_ctc_out=None):
        super(PostLowEnergyv2, self).__init__()
        #TODO: speake_emb will be removed
        ##def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=True):
        # Add Oct. 16
        self.gender_emb = gender_emb
        self.speaker_emb = speaker_emb
        self.concat = concat
        self.vq_code_flag = hp.vq_code
        self.spk_emb_postprocess_type = spk_emb_postprocess_type
        self.layers_ctc_out = layers_ctc_out
        self.version = hp.version

        if self.concat:
            out_dim = vocab_size + d_model
            if  spk_emb_dim is not None and spk_emb_postprocess_type is not None:
                out_dim += spk_emb_dim
        else:
            self.linear1 = nn.Linear(vocab_size, d_model)
            self.linear2 = nn.Linear(d_model, d_model)
            if spk_emb_postprocess_type == 'speaker_id':
                self.linear_xvector = nn.Embedding(spk_emb_dim, d_model)
            elif spk_emb_postprocess_type == 'x_vector':
                self.linear_xvector = nn.Linear(spk_emb_dim, d_model)
                #self.layer_norm_xvector = nn.LayerNorm(d_model)
            out_dim = d_model

        if self.vq_code_flag:
            #self.vq_encoder_sp = nn.Embedding(247, out_dim) 
            #self.layer_norm = nn.LayerNorm(out_dim)
            self.vq_encoder_lmfb = nn.Conv1d(vocab_size, out_dim, 1)
            #self.quantize_sp = Quantize(out_dim, 15) # n_embed
            self.quantize_lmfb = Quantize(out_dim, 20) # n_embed
            #OLD
            #self.vq_encoder = nn.Conv1d(out_dim, out_dim, 1)
            #self.quantize_sp = Quantize(out_dim, 15) # n_embed
            #self.quantize = Quantize(out_dim, 15) # n_embed

        #if self.gender_emb or self.speaker_emb or self.ctc_flag:
        #    self.encoder = EncoderPostprocessing(out_dim, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, multi_speaker=False, spk_emb_dim=None, embedding=embedding, gender_emb=gender_emb, speaker_emb=speaker_emb, ctc_out=ctc_out)
        #else
        if hp.post_conformer:
            self.encoder = ConformerEncoder(out_dim, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, embedding=embedding)
        else:
            self.encoder = Encoder(out_dim, d_model, N, heads, ff_conv_kernel_size, concat_after_encoder, dropout, embedding=embedding, layers_ctc_out=layers_ctc_out)

        self.out = nn.Linear(d_model, out_size)

        if self.version == 8:
            self.out_replace = nn.Linear(d_model, out_size)

    def forward(self, src, src_mask, variance_adaptor_output, spkr_emb=None, gender=None):
        if self.concat:
            input_ = torch.cat((src, variance_adaptor_output), dim=-1)
            if self.spk_emb_postprocess_type is not None:
                input_ = torch.cat((input_, spkr_emb.unsqueeze(1).expand(-1, input_.shape[1], -1)), dim=-1)
        else:
            input_ = self.linear1(src) + self.linear2(variance_adaptor_output)
            if self.spk_emb_postprocess_type is not None:
                input_ = input_ + self.linear_xvector(spkr_emb).unsqueeze(1)
                #input_ = input_ + self.layer_norm_xvector(self.linear_xvector(spkr_emb).unsqueeze(1))

        if self.vq_code_flag:
            # OLD
            vq_input_lmfb = self.vq_encoder_lmfb(src.transpose(1,2))
            #quantize_sp, diff_sp, embed_ind_sp = self.quantize_sp(vq_input_)
            quantize_lmfb, diff_lmfb, embed_ind_lmfb = self.quantize_lmfb(vq_input_lmfb, mean=True)
            print(embed_ind_lmfb)
            print(f'diff_lmfb={diff_lmfb.item()}',end=',')
            input_ = input_  + quantize_lmfb.unsqueeze(1) #+ quantize_sp.unsqueeze(1) + quantize_lmfb.unsqueeze(1)
            diff = diff_lmfb #diff_sp + diff_lmfb
            
            # TODO: make feature normalize for time sequence
            #vq_input_ = self.vq_encoder_sp(spkr_emb)
            #vq_input_ = self.layer_norm(vq_input_)
            #vq_input_lmfb = self.vq_encoder_lmfb(input_.transpose(1,2))
            #quantize_sp, diff_sp, embed_ind_sp = self.quantize_sp(vq_input_)
            #quantize_lmfb, diff_lmfb, embed_ind_lmfb = self.quantize_lmfb(vq_input_lmfb, mean=True)
            #input_ = input_  + quantize_lmfb.unsqueeze(1) #+ quantize_sp.unsqueeze(1) + quantize_lmfb.unsqueeze(1)
            #print(f'diff_lmfb={diff_lmfb.item()}',end=',')
            #print(f'diff_sp={diff_sp.item()}, diff_lmfb={diff_lmfb.item()}',end=',')
            #print(f'embed_ind_sp={embed_ind_sp.item()}', end=',')
            #print(f'embed_ind_lmfb={embed_ind_lmfb.item()}', end=',')
            #diff = diff_lmfb #diff_sp + diff_lmfb

            # hierar nan???/
            #vq_input_ = self.vq_encoder_sp(spkr_emb)
            #quantize_sp, diff_sp, embed_ind = self.quantize_sp(vq_input_)
            #input_ = input_ + quantize_sp.unsqueeze(1)
            #vq_input_lmfb = self.vq_encoder_lmfb(input_.transpose(1,2))
            #quantize_lmfb, diff_lmfb, embed_ind = self.quantize_lmfb(vq_input_lmfb, mean=True)
            #input_ = input_ + quantize_lmfb.unsqueeze(1)
            #print('diff_sp', diff_sp)
            #print('diff_lmfb', diff_lmfb)
            #diff = diff_sp + diff_lmfb
            # hi 2 sp2 lmfb 1
            #vq_input_lmfb = self.vq_encoder_lmfb(input_.transpose(1,2))
            #quantize_lmfb, diff_lmfb, embed_ind = self.quantize_lmfb(vq_input_lmfb, mean=True)
            #input_ = input_ + quantize_lmfb.unsqueeze(1)
            #vq_input_ = self.vq_encoder_sp(spkr_emb)
            #quantize_sp, diff_sp, embed_ind = self.quantize_sp(vq_input_)
            #input_ = input_ + quantize_sp.unsqueeze(1)
            #print('diff_lmfb', diff_lmfb)
            #print('diff_sp', diff_sp)
            #diff = diff_sp + diff_lmfb
        else:
            diff = None

        #if self.gender_emb or self.speaker_emb or self.ctc_flag:
        #    e_outputs, ctc_out, attn_enc = self.encoder(input_, src_mask, spkr_emb, gender=gender)
        #else:
        if self.layers_ctc_out:
            e_outputs, attn_enc, ctc_outs = self.encoder(input_, src_mask, spkr_emb)
        else:
            e_outputs, attn_enc = self.encoder(input_, src_mask, spkr_emb)
            ctc_outs = None

        outputs = self.out(e_outputs)
        if self.version == 8:
            outputs_replace = self.out_replace(e_outputs)
            outputs = (outputs, outputs_replace)
        
        return outputs, ctc_outs, diff

class Quantize(nn.Module):
    def __init__(self, embeded_dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = embeded_dim
        self.n_embed = n_embed
        embed = torch.randn(embeded_dim, n_embed)
        self.decay = decay
        self.eps = eps

        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input_, mean=False):
        # later, we consider
        if mean:
            input_ = input_.mean(dim=2)
        flatten = input_.reshape(-1, self.dim)

        dist = (flatten.pow(2).sum(-1, keepdim=True) -2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True))
        
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        #embed_ind = embed_ind.view(*input_[:-1].shape)

        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1-self.decay)

            self.embed_avg.data.mul_(self.decay).data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n)
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input_).pow(2).mean()
        quantize = input_ + (quantize - input_).detach()

        return quantize, diff, embed_ind
