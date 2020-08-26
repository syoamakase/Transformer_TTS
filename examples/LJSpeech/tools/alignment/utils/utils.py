# -*- coding: utf-8 -*-
import copy
import numpy as np
import os
import sys
from struct import unpack, pack
import torch
import torch.nn as nn

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

def frame_stacking(x, x_lengths, stack):
    batch_size = x.shape[0]
    newlen = x.shape[1] // stack
    x_lengths = x_lengths // stack
    stacked_x = x[:, 0:newlen*stack].reshape(batch_size, newlen, hp.lmfb_dim * stack)
    return stacked_x, x_lengths

def load_model(model_file):
    """
    To load PyTorch models either of single-gpu and multi-gpu based model
    """
    model_state = torch.load(model_file)
    is_multi_loading = True if torch.cuda.device_count() > 1 else False
    # This line may include bugs!!
    is_multi_loaded = True if 'module' in list(model_state.keys())[0] else False

    if is_multi_loaded is is_multi_loading:
        return model_state

    # the model to load is multi-gpu and the model to use is single-gpu
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
    else:
        print('ERROR in load model')
        sys.exit(1)

def overwrite_hparams(args):
    for key, value in args._get_kwargs():
        if value is not None and value != 'load_name':
            setattr(hp, key, value) 

def fill_variables():
    if hasattr(hp, 'num_hidden_nodes'):
        num_hidden_nodes_encoder = hp.num_hidden_nodes
        num_hidden_nodes_decoder = hp.num_hidden_nodes
    else:
        num_hidden_nodes_encoder = 512
        num_hidden_nodes_decoder = 512
        
    default_var = {'spm_model':None, 'T_norm':True, 'B_norm':False, 'save_per_epoch':1, 'lr_adjust_epoch': 20,
                   'reset_optimizer_epoch': 40, 'num_hidden_nodes_encoder':num_hidden_nodes_encoder, 'num_hidden_nodes_decoder':num_hidden_nodes_decoder,
                    'comment':'', 'load_name_lm':None, 'shuffle': False, 'num_mask_F':1, 'num_mask_T':1,
                    'max_width_F':23, 'max_width_T':100}
    for key, value in default_var.items():
        if not hasattr(hp, key):
            # print('{} is not found in hparams. defalut {} is used.'.format(key, value))
            setattr(hp, key, value)
