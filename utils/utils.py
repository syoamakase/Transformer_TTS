# -*- coding: utf-8 -*-
import copy
import numpy as np
from struct import unpack, pack
import os
import sys
import torch
import torch.nn as nn
from shutil import copyfile

from utils import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_config():
    print(f'PID = {os.getpid()}')
    print(f'PyTorch version = {torch.__version__}')
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print('cuda device = {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    for key in hp.__dict__.keys():
        if not '__' in key:
            print('{} = {}'.format(key, eval('hp.'+key)))

def load_dat(filename):
    """
    To read binary data in htk file.
    The htk file includes log mel-scale filter bank.

    Args:
        filename : file name to read htk file

    Returns:
        dat : (log mel-scale filter bank dim) x (time frame)

    """
    fh = open(filename, "rb")
    spam = fh.read(12)
    _, _, sampSize, _ = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat

def frame_stacking(x, x_lengths, stack):
    batch_size = x.shape[0]
    newlen = x.shape[1] // stack
    x_lengths = x_lengths // stack
    stacked_x = x[:, 0:newlen*stack].reshape(batch_size, newlen, hp.lmfb_dim * stack)
    return stacked_x, x_lengths

def frame_stacking_legacy(x, stack):
    newlen = len(x) // stack
    stacked_x = x[0:newlen*stack].reshape(newlen, hp.lmfb_dim * stack)
    return stacked_x

def onehot(labels, num_output):
    """
    To make onehot vector.
    ex) labels : 3 -> [0, 0, 1, 0, ...]

    Args:
        labels : true label ID
        num_output : the number of entry

    Returns:
        utt_label : one hot vector.
    """
    utt_label = np.zeros((len(labels), num_output), dtype='float32')
    for i in range(len(labels)):
        utt_label[i][labels[i]] = 1.0
    return utt_label

def load_model(model_file):
    """
    To load the both of multi-gpu model and single gpu model.
    """
    model_state = torch.load(model_file)
    is_multi_loading = True if torch.cuda.device_count() > 1 else False
    # This line may include bugs!!
    is_multi_loaded = True if 'module' in list(model_state.keys())[0] else False

    if is_multi_loaded is is_multi_loading:
        return model_state

    elif is_multi_loaded is False and is_multi_loading is True:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state['module.'+key] = model_state[key]

        return new_model_state

    elif is_multi_loaded is True and is_multi_loading is False:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]

        return new_model_state

def init_weight(m):
    """
    To initialize weights and biases.
    """
    classname = m.__class__.__name__
    if classname.find('linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

def adjust_learning_rate(optimizer, epoch):
    if hp.reset_optimizer_epoch is not None:
        if (epoch % hp.reset_optimizer_epoch) > hp.lr_adjust_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8 
    else:
        if epoch > hp.lr_adjust_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8

def overwrite_hparams(args):
    for key, value in args._get_kwargs():
        if value is not None and value != 'load_name':
            setattr(hp, key, value) 

def fill_variables():
    default_var = {'spm_model':None, 'mean_file':None, 'var_file': None, 'log_dir': 'logs', 'CTC_training': False}
    for key, value in default_var.items():
        if not hasattr(hp, key):
            print('{} is not found in hparams. defalut {} is used.'.format(key, value))
            setattr(hp, key, value)

def get_learning_rate(step, d_model, warmup_factor, warmup_step):
    return warmup_factor * min(step ** -0.5, step * warmup_step ** -1.5) * (d_model ** -0.5)

def guide_loss(self, alignments, text_lengths, mel_lengths):
    B, n_layers, n_heads, T, L = alignments.size()
    
    # B, T, L
    W = alignments.new_zeros(B, T, L)
    mask = alignments.new_zeros(B, T, L)
    
    for i, (t, l) in enumerate(zip(mel_lengths, text_lengths)):
        mel_seq = alignments.new_tensor( torch.arange(t).to(torch.float32).unsqueeze(-1)/t )
        text_seq = alignments.new_tensor( torch.arange(l).to(torch.float32).unsqueeze(0)/l )
        x = torch.pow(mel_seq-text_seq, 2)
        W[i, :t, :l] += alignments.new_tensor(1-torch.exp(-3.125*x))
        mask[i, :t, :l] = 1
    
    # Apply guided_loss to 2 heads of the last 2 layers 
    applied_align = alignments[:, -2:, :2]
    losses = applied_align*(W.unsqueeze(1).unsqueeze(1))
    
    return torch.mean(losses.masked_select(mask.unsqueeze(1).unsqueeze(1).to(torch.bool)))
