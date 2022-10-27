from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from Models.modules import SQEmbedding

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args

def repeat(N, fn):
    """Repeat module N times.
    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn() for _ in range(N)])

class SQVarianceAdaptor(nn.Module):
    ## Where do f0_min and f0_max come??? 
    def __init__(self, d_model_encoder, n_bins=256, f0_min=71.0, f0_max=795.8, energy_min=0.0, energy_max=315.0,
                 log_offset=1., pitch_pred=True, energy_pred=True, dropout=0.5):
        super().__init__()

        # def __init__(self, input_size, filter_size, kernel_size, dropout):

        self.pitch_pred = pitch_pred
        self.energy_pred = energy_pred

        # encoder_hidden_size, variance_predictor_filter_size=256, variance_predictor_kernel_size=3, variance_predictor_dropout=0.5
        self.param_var_q = "gaussian_1" # hp.param_var_q
        n_embeddings = 128 #64 # d_model_encoder
        embedding_dim = 384 #64
        self.embedding_dim = embedding_dim
        log_var_q_scalar = torch.Tensor(1)
        log_var_q_scalar.fill_(10.0).log_()
        self.register_parameter("log_var_q_scalar", nn.Parameter(log_var_q_scalar))
        self.codebook = SQEmbedding(self.param_var_q, n_embeddings, embedding_dim)

        self.duration_predictor = VariancePredictor(d_model_encoder, variance_predictor_dropout=dropout)
        self.length_regulator = LengthRegulator()
        if self.pitch_pred:
            self.pitch_predictor = VariancePredictor(d_model_encoder, variance_predictor_dropout=dropout)
            self.pitch_bins = torch.exp(torch.linspace(np.log(f0_min), np.log(f0_max), n_bins-1)).to(DEVICE)
            self.pitch_embedding = nn.Embedding(n_bins, d_model_encoder)

        if self.energy_pred:
            self.energy_predictor = VariancePredictor(d_model_encoder, variance_predictor_dropout=dropout)
            self.energy_bins = torch.linspace(energy_min, energy_max, n_bins-1).to(DEVICE)
            self.energy_embedding = nn.Embedding(n_bins, d_model_encoder)

        self.log_offset = 1.

    def forward(self, x, src_mask, mel_mask=None, duration_target=None, 
                pitch_target=None, energy_target=None, max_len=None, temperature=None):
        
        assert (self.training is True and temperature is not None) or (self.training is False and temperature is None)
        if self.param_var_q == "gaussian_1":
            log_var_q = self.log_var_q_scalar
        elif self.param_var_q == "gaussian_3" or self.param_var_q == "gaussian_4":
            log_var_q = z[:, :, self.embedding_dim:] + self.log_var_q_scalar
        else:
            raise Exception("Undefined param_var_q")

        z = x[:, :, :self.embedding_dim]
        if self.training:
            z, sq_vae_loss, sq_vae_perplexity, indices = self.codebook(z, log_var_q, temperature)
        else:
            z, indices = self.codebook.encode(z, log_var_q)
            sq_vae_loss = None
            sq_vae_perplexity = None

        log_duration_prediction = self.duration_predictor(z, src_mask)
        # import pdb; pdb.set_trace()
        if duration_target is not None:
            if mel_mask is not None:
                max_len = mel_mask.shape[2]
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            if mel_mask is not None:
                max_len = mel_mask.shape[2]
            duration_rounded = torch.clamp(torch.round(torch.exp(log_duration_prediction)-self.log_offset), min=0)

            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            z, mel_len = self.length_regulator(z, duration_rounded, max_len)
            x = x + z
            # TODO: get_mask_from_lengths
            if not self.training:
                mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_pred:
            pitch_prediction = self.pitch_predictor(x, mel_mask)
            if pitch_target is not None:
                pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_target, self.pitch_bins))
            else:
                pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_prediction, self.pitch_bins))
        else:
            pitch_prediction = None

        if self.energy_pred:
            energy_prediction = self.energy_predictor(x, mel_mask)
            if energy_target is not None:
                energy_embedding = self.energy_embedding(torch.bucketize(energy_target, self.energy_bins))
            else:
                energy_embedding = self.energy_embedding(torch.bucketize(energy_prediction, self.energy_bins))
        else:
            energy_prediction = None

        text_dur_predicted = x
        if self.pitch_pred:
            x = x + pitch_embedding
        if self.energy_pred:
            x = x + energy_embedding
        # x = x + pitch_embedding + energy_embedding

        return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask, text_dur_predicted, sq_vae_loss, sq_vae_perplexity


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_length=None):
        output = list()
        mel_pos = list()

        for batch, expand_target in zip(x, duration):
            output.append(self.expand(batch, expand_target))
            mel_pos.append(torch.arange(1, len(output[-1])+1).to(DEVICE))
         
        if max_length is not None:
            output = pad(output, max_length)
            mel_pos = pad(output, max_length)
        else:
            output = pad(output)
            mel_pos = pad(mel_pos)

        return output, mel_pos

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_length=None):
        if self.use_lstm:
            output, mel_pos = self.LR(x, duration, max_length)
        else:
            output, mel_pos = self.LR(x, duration, max_length)
        return output, mel_pos

class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, encoder_hidden_size, variance_predictor_filter_size=256, variance_predictor_kernel_size=3, variance_predictor_dropout=0.5):
        super(VariancePredictor, self).__init__()

        self.input_size = encoder_hidden_size
        self.filter_size = variance_predictor_filter_size
        self.kernel = variance_predictor_kernel_size
        self.conv_output_size = variance_predictor_filter_size
        self.dropout = variance_predictor_dropout

        self.conv1 = nn.Conv1d(self.input_size, self.filter_size,
                              kernel_size=self.kernel, padding=1)
        self.relu1 = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(self.filter_size)
        self.dropout1 = nn.Dropout(self.dropout)
        self.conv2 = nn.Conv1d(self.filter_size, self.filter_size,
                               kernel_size=self.kernel, padding=1)
        self.relu2 = nn.ReLU()
        self.layer_norm2 = nn.LayerNorm(self.filter_size)
        self.dropout2 = nn.Dropout(self.dropout)

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        x = self.conv1(encoder_output.transpose(1,2))
        x = self.relu1(x).transpose(1,2)
        x = self.dropout1(self.layer_norm1(x)).transpose(1,2)
        x = self.conv2(x)
        x = self.relu2(x).transpose(1,2)
        x = self.dropout2(self.layer_norm2(x))
        out = self.linear_layer(x)
        out = out.squeeze(-1)
        
        if not self.training and out.dim() == 1:
            out = out.unsqueeze(0)
        if mask is not None:
            out = out.masked_fill(mask.squeeze(1)==0, 0.)

        return out

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for _, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(DEVICE)
    mask = ids <= lengths

    return mask

class MelEncoder(nn.Module):
    """"""

    def __init__(self, encoder_hidden_size, variance_predictor_filter_size=256, variance_predictor_kernel_size=3, variance_predictor_dropout=0.5):
        super(VariancePredictor, self).__init__()

        self.input_size = encoder_hidden_size
        self.filter_size = variance_predictor_filter_size
        self.kernel = variance_predictor_kernel_size
        self.conv_output_size = variance_predictor_filter_size
        self.dropout = variance_predictor_dropout

        self.conv1 = nn.Conv1d(self.input_size, self.filter_size,
                              kernel_size=self.kernel, padding=1)
        self.relu1 = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(self.filter_size)
        self.dropout1 = nn.Dropout(self.dropout)
        self.conv2 = nn.Conv1d(self.filter_size, self.filter_size,
                               kernel_size=self.kernel, padding=1)
        self.relu2 = nn.ReLU()
        self.layer_norm2 = nn.LayerNorm(self.filter_size)
        self.dropout2 = nn.Dropout(self.dropout)

        # self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        x = self.conv1(encoder_output.transpose(1,2))
        x = self.relu1(x).transpose(1,2)
        x = self.dropout1(self.layer_norm1(x)).transpose(1,2)
        x = self.conv2(x)
        x = self.relu2(x).transpose(1,2)
        out = self.dropout2(self.layer_norm2(x))
        # out = self.linear_layer(x)
        # out = out.squeeze(-1)
        
        if not self.training and out.dim() == 1:
            out = out.unsqueeze(0)
        if mask is not None:
            out = out.masked_fill(mask.squeeze(1)==0, 0.)

        return out