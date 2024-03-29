from collections import OrderedDict
from pydoc import doc
from unicodedata import bidirectional
from unittest import result
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

from Models.modules import PositionalEncoder

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

class VarianceAdaptor(nn.Module):
    ## Where do f0_min and f0_max come??? 
    def __init__(self, d_model_encoder, n_bins=256, f0_min=71.0, f0_max=795.8, energy_min=0.0, energy_max=315.0,
                log_offset=1., pitch_pred=True, energy_pred=True, dropout=0.5, use_rnn_length=False, use_pos=False):
        super().__init__()

        # def __init__(self, input_size, filter_size, kernel_size, dropout):

        self.pitch_pred = pitch_pred
        self.energy_pred = energy_pred
        self.use_rnn_length = use_rnn_length
        self.use_pos = use_pos

        if use_pos:
            self.pos = PositionalEncoder(d_model_encoder)

        self.duration_predictor = VariancePredictor(d_model_encoder, variance_predictor_dropout=dropout)
        self.length_regulator = LengthRegulator()
        if self.use_rnn_length:
            self.rnn_length = nn.LSTM(d_model_encoder, d_model_encoder, batch_first=True, bidirectional=False)
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
                pitch_target=None, energy_target=None, max_len=None, p_scheduled_sampling=0.0, pitch_perturbation=False, duration_perturbation=False):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if duration_target is not None:
            if mel_mask is not None:
                max_len = mel_mask.shape[2]
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(torch.round(torch.exp(log_duration_prediction)-self.log_offset), min=0)
            if duration_perturbation:
                duration_rand_list = [0.8, 0.9, 1.0, 1.1, 1.2]
                rand_weight = random.sample(duration_rand_list, 1)[0]
                print('dur', rand_weight)
                
                duration_rounded = torch.round(duration_rounded * rand_weight)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            # TODO: get_mask_from_lengths
            mel_mask = get_mask_from_lengths(mel_len)

        if self.use_pos:
            x = self.pos(x)
            # import pdb; pdb.set_trace()

        if self.use_rnn_length:
            # import pdb; pdb.set_trace()
            x, (_,_) = self.rnn_length(x)

        if self.pitch_pred:
            pitch_prediction = self.pitch_predictor(x, mel_mask)
            if pitch_target is not None:
                # import pdb; pdb.set_trace()
                pitch_target_scheduled = scheduled_sampling(predicted=pitch_prediction, target=pitch_target, p=p_scheduled_sampling)
                
                pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_target_scheduled, self.pitch_bins))
            else:
                # pitch_prediction = 1.4 * pitch_prediction
                if pitch_perturbation:
                    pitch_rand_list = [0.8, 0.9, 1.0, 1.1, 1.2]
                    # import pdb; pdb.set_trace()
                    rand_weight = random.sample(pitch_rand_list, 1)[0]
                    pitch_prediction = rand_weight * pitch_prediction

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

        text_dur_predicted = x #copy.deepcopy(x)
        if self.pitch_pred:
            x = x + pitch_embedding
        if self.energy_pred:
            x = x + energy_embedding
        # x = x + pitch_embedding + energy_embedding

        return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask, text_dur_predicted

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, use_lstm=False, hidden_dim=None):
        super(LengthRegulator, self).__init__()
        self.use_lstm = use_lstm
        if self.use_lstm:
            assert hidden_dim is not None
            self.rnn = nn.LSTMCell(hidden_dim, hidden_dim)

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

    # def LR_with_LSTM(self, x, duration, max_length=None):
    #     output = list()
    #     mel_pos = list()

    #     for batch, expand_target in zip(x, duration):
    #         for _ in expand_target:
    #             hx, cx = rnn(batch[i], (hx, cx))
    #             output.append(hx)
    #         # output.append(self.expand(batch, expand_target))
    #             mel_pos.append(torch.arange(1, len(output[-1])+1).to(DEVICE))

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

        # self.input_size = hp.encoder_hidden
        # self.filter_size = hp.variance_predictor_filter_size
        # self.kernel = hp.variance_predictor_kernel_size
        # self.conv_output_size = hp.variance_predictor_filter_size
        # self.dropout = hp.variance_predictor_dropout
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

def scheduled_sampling(predicted, target, p):
    """ Scheduled sampling for applying to pitch (and energy).
    TODO: apply prob to phone-level ?, batch-wise?

    Args:
        predicted (FloatTesnsor): (B x T). Predicted acoustic information
        target (FloatTensor): (B x T). Ground-truth acoustic information
        p (float): Propability to replace true target into predicted. 
    """
    if p == 0.0:
        # import pdb; pdb.set_trace()
        return target
        
    assert predicted.shape == target.shape

    result = target.clone()
    rand_vals = torch.rand(predicted.shape[0])
    for i in range(predicted.shape[0]):
        if rand_vals[i] < p:
            result[i] = predicted[i]
    # import pdb; pdb.set_trace()
    return result
