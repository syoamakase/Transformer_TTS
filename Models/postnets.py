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
    def __init__(self, num_hidden, mel_dim, reduction_rate):
        """
        :param num_hidden: dimension of hidden 
        """
        super(PostConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=mel_dim*reduction_rate,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=4)
        self.conv_list = clones(nn.Conv1d(in_channels=num_hidden,
                                     out_channels=num_hidden,
                                     kernel_size=5,
                                     padding=4), 3)
        self.conv2 = nn.Conv1d(in_channels=num_hidden,
                          out_channels=mel_dim*reduction_rate,
                          kernel_size=5,
                          padding=4)

        # self.out = nn.Linear(num_hidden, mel_dim)
        self.out = nn.Linear(num_hidden, mel_dim*reduction_rate)

        self.batch_norm_list = clones(nn.BatchNorm1d(num_hidden), 3)
        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=0.5) for _ in range(3)])

    def forward(self, input_, mask=None):
        mel_pred = self.out(input_).transpose(1,2)
        # Causal Convolution (for auto-regressive)
        input_ = self.dropout1(torch.tanh(self.pre_batchnorm(self.conv1(mel_pred)[:, :, :-4])))
        for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
            input_ = dropout(torch.tanh(batch_norm(conv(input_)[:, :, :-4])))
        input_ = self.conv2(input_)[:, :, :-4]
        input_ = mel_pred + input_
        return mel_pred.transpose(1,2), input_.transpose(1,2)
