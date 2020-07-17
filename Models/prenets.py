
#-*- coding: utf-8 -*-
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderPreNet(nn.Module):
    """
    Prenet before passing through the network
    """
    def __init__(self, input_size, output_size, hidden_size=256, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(DecoderPreNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(p)),
             ('fc2', nn.Linear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(p)),
        ]))

    def forward(self, input_):
        out = self.layer(input_)

        return out

class EncoderPreNet(nn.Module):
    # copy
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.conv_1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.batch_norm_1 = nn.BatchNorm1d(d_model)
        self.conv_2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.batch_norm_2 = nn.BatchNorm1d(d_model)
        self.conv_3 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.batch_norm_3 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        # self.final_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.embed(x)
        x = self.conv_1(x.transpose(1,2))
        x = self.dropout1(torch.relu(self.batch_norm_1(x)))
        x = self.conv_2(x)
        x = self.dropout2(torch.relu(self.batch_norm_2(x)))
        x = self.conv_3(x)
        x = self.dropout3(torch.relu(self.batch_norm_3(x)))

        return self.final_out(x.transpose(1,2))
        # return x.transpose(1,2)