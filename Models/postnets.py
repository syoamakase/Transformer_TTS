#-*- coding: utf-8 -*-
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PostConvNet(nn.Module):
    """
    Post Convolutional Network (mel --> mel)
    """
    def __init__(self, num_hidden, mel_dim, reduction_rate, dropout=0.5, output_type=None, num_group=None):
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
            # ignore
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
            mel_pred = self.out(input_).transpose(1, 2)
            # Causal Convolution (for auto-regressive)
            input_ = self.dropout1(torch.tanh(self.pre_batchnorm(self.conv1(mel_pred)[:, :, :-4])))
            for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
                input_ = dropout(torch.tanh(batch_norm(conv(input_)[:, :, :-4])))
            input_ = self.conv2(input_)[:, :, :-4]
            input_ = mel_pred + input_
            return mel_pred.transpose(1, 2), input_.transpose(1, 2)
