# -*- coding: utf-8 -*-
import argparse
import copy
from collections import OrderedDict
import itertools
import os
from struct import unpack, pack
import sys

import matplotlib.pyplot as plt
import numpy as np
import math
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm
from datasets import datasets_transformer
from torch.utils.data import DataLoader

from utils import hparams as hp
from utils.utils import log_config, fill_variables

from Models.transformer import Transformer

random.seed(77)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            new_model_state['module.' + key] = model_state[key]

        return new_model_state

    elif is_multi_loaded is True and is_multi_loading is False:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]

        return new_model_state

def nopeak_mask(size):
    """
    npeak_mask(4)
    >> tensor([[[ 1,  0,  0,  0],
         [ 1,  1,  0,  0],
         [ 1,  1,  1,  0],
         [ 1,  1,  1,  1]]], dtype=torch.uint8)

    """
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).to(DEVICE)
    return np_mask

def create_masks(src_pos, trg_pos, src_pad=0, trg_pad=0):
    src_mask = (src_pos != src_pad).unsqueeze(-2)

    if trg_pos is not None:
        trg_mask = (trg_pos != trg_pad).unsqueeze(-2)
        size = trg_pos.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        if trg_pos.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_name', required=True)
    parser.add_argument('--test_script', default=None)
    args = parser.parse_args()
    load_name = args.load_name

    if os.path.exists(os.path.join(os.path.dirname(load_name), 'hparams.py')):
        hp_file = os.path.join(os.path.dirname(load_name), 'hparams.py')

    hp.configure(hp_file)
    fill_variables(hp)
    save_path = os.path.join(os.path.dirname(load_name), 'dev')
    os.makedirs(save_path, exist_ok=True)

    # initialize pytorch
    # model = Transformer(src_vocab=hp.vocab_size, trg_vocab=hp.mel_dim, N_e=6, N_d=6, heads=8, d_model=384, dropout=hp.dropout)
    model = Transformer(hp=hp, src_vocab=hp.vocab_size, trg_vocab=hp.mel_dim,
                        d_model_encoder=hp.d_model_encoder, N_e=hp.n_layer_encoder, n_head_encoder=hp.n_head_encoder,
                        ff_conv_kernel_size_encoder=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.concat_after_encoder,
                        d_model_decoder=hp.d_model_decoder, N_d=hp.n_layer_decoder, n_head_decoder=hp.n_head_decoder,
                        ff_conv_kernel_size_decoder=hp.ff_conv_kernel_size_decoder, concat_after_decoder=hp.concat_after_decoder,
                        reduction_rate=hp.reduction_rate, dropout=0.0, dropout_prenet=0.0, dropout_postnet=0.0,
                        multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim, spk_emb_architecture=hp.spk_emb_architecture)

    model.to(DEVICE)
    model.eval()

    model.load_state_dict(load_model(f"{load_name}"))
    if args.test_script is not None:
        test_script = args.test_script
    else:
        test_script = hp.test_script
    dataset_test = datasets_transformer.TestDatasets(test_script)
    collate_fn_transformer = datasets_transformer.collate_fn_test
    sampler = datasets_transformer.NumBatchSampler(dataset_test, 1, shuffle=False)#hp.batch_size)

    dataloader = DataLoader(dataset_test, batch_sampler=sampler, num_workers=1, collate_fn=collate_fn_transformer)

    mean_value = np.load(hp.mean_file).reshape(-1, hp.mel_dim)
    var_value = np.load(hp.var_file).reshape(-1, hp.mel_dim)

    for idx, d in enumerate(dataloader): 
        text, mel, pos_text, text_lengths, spk_emb = d
        # torch.LongTensor(text), mel_output, torch.LongTensor(pos_text), torch.LongTensor(text_length), spk_emb

        text = text.to(DEVICE)
        #mel = mel.to(DEVICE)
        pos_text = pos_text.to(DEVICE)
        #pos_mel = pos_mel.to(DEVICE)
        text_lengths = text_lengths.to(DEVICE)
        #stop_token = stop_token.to(DEVICE)
        spk_emb = spk_emb.to(DEVICE)
        #spk_emb = F.normalize(spk_emb)

        #batch_size = mel.shape[0]
        dummy_input = torch.zeros((1, 1, 80), dtype=torch.float, device=DEVICE)

        pos_mel = torch.arange(start=1, end=2, dtype=torch.long, device=DEVICE).unsqueeze(0)
        src_mask, trg_mask = create_masks(pos_text, pos_mel)
        mel_input = dummy_input
        outputs_prenet, outputs_postnet, outputs_stop_token, attn_enc, attn_dec_dec, attn_dec_enc = model(text, mel_input, src_mask, trg_mask, spk_emb, training=False)

        if hp.reduction_rate >= 1:
            b, t, c = outputs_prenet.shape
            outputs_prenet = outputs_prenet.reshape(b, t*hp.reduction_rate, int(c//hp.reduction_rate))
            outputs_postnet = outputs_postnet.reshape(b, t*hp.reduction_rate, int(c//hp.reduction_rate))
            mel_input = outputs_postnet[:, -1, :].unsqueeze(1)

        mel_input = torch.cat((dummy_input, mel_input), dim=1)
        case = 0
        with torch.no_grad():
            for i in range(2, 500):
                # mel_input = torch.cat((dummy_input, mel_input), dim=1)
                pos_mel = torch.arange(start=1,end=i+1,dtype=torch.long, device=DEVICE).unsqueeze(0)
                src_mask, trg_mask = create_masks(pos_text, pos_mel)

                print(f'{i}')

                outputs_prenet, outputs_postnet, outputs_stop_token, attn_enc, attn_dec_dec, attn_dec_enc = model(text, mel_input, src_mask, trg_mask, spk_emb, training=False)

                if case == 1:
                    mel_input = torch.cat((mel_input, outputs_postnet[:, -1, :].unsqueeze(1)), dim=1) 
                else:
                    # mel_input = outputs_postnet
                    if hp.reduction_rate >= 1:
                        b,t,c = outputs_prenet.shape
                        outputs_prenet = outputs_prenet.reshape(b, t*hp.reduction_rate, int(c//hp.reduction_rate))
                        outputs_postnet = outputs_postnet.reshape(b, t*hp.reduction_rate, int(c//hp.reduction_rate))
                        mel_input = outputs_postnet[:,::hp.reduction_rate, :]
                    mel_input = torch.cat((dummy_input, mel_input), dim=1)
                # print(mel_input.shape)
                if hp.reduction_rate > 1:
                    if torch.sigmoid(outputs_stop_token).mean(dim=-1)[0, -1] > 0.5:
                        break
                else:
                    # import pdb; pdb.set_trace()
                    if torch.sigmoid(outputs_stop_token)[0, -1] > 0.5:
                        break
            outputs = outputs_postnet[0].cpu().detach().numpy()
            outputs *= np.sqrt(var_value)
            outputs += mean_value
            import pdb; pdb.set_trace()
            output_name = os.path.join(save_path, str(idx)+'.npy')
            print(f'save {output_name}')
            np.save(output_name, outputs)
            sys.stdout.flush()
