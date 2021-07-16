# -*- coding: utf-8 -*-
import argparse
import os
from struct import unpack, pack
import sys

import matplotlib.pyplot as plt
import numpy as np
import math
import random

import torch
from torch.autograd import Variable

from tqdm import tqdm
import datasets.datasets_fastspeech2_dev as datasets
from torch.utils.data import DataLoader

from utils import hparams as hp
from utils.utils import log_config, fill_variables

from Models.fastspeech2 import FastSpeech2

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
            new_model_state['module.'+key] = model_state[key]

        return new_model_state

    elif is_multi_loaded is True and is_multi_loading is False:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]

        return new_model_state

def npeak_mask(size):
    """
    npeak_mask(4)
    >> tensor([[[ 1,  0,  0,  0],
         [ 1,  1,  0,  0],
         [ 1,  1,  1,  0],
         [ 1,  1,  1,  1]]], dtype=torch.uint8)

    """
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0).to(DEVICE)
    return np_mask

def create_masks(src_pos, trg_pos, task='transformer', src_pad=0, trg_pad=0):
    src_mask = (src_pos != src_pad).unsqueeze(-2)
    if task.lower() == 'fastspeech2':
        trg_mask = (trg_pos != trg_pad).unsqueeze(-2)
    else:
        # For general transformer
        if trg_pos is not None:
            trg_mask = (trg_pos != trg_pad).unsqueeze(-2)
            size = trg_pos.size(1) # get seq_len for matrix
            np_mask = npeak_mask(size)
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
    test_script = args.test_script

    if os.path.exists(os.path.join(os.path.dirname(load_name), 'hparams.py')):
        hp_file = os.path.join(os.path.dirname(load_name), 'hparams.py')

    hp.configure(hp_file)
    fill_variables(hp)
    epoch = os.path.basename(load_name).replace('network.average_', '')
    save_path = os.path.join(os.path.dirname(load_name), 'dev/'+epoch)
    os.makedirs(save_path, exist_ok=True)

    if test_script is not None:
        hp.test_script = test_script
    else:
        hp.test_script = hp.train_script

    model = FastSpeech2(hp=hp, src_vocab=hp.vocab_size, trg_vocab=hp.mel_dim, d_model_encoder=hp.d_model_encoder, N_e=hp.n_layer_encoder,
                        n_head_encoder=hp.n_head_encoder, ff_conv_kernel_size_encoder=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.ff_conv_kernel_size_encoder,
                        d_model_decoder=hp.d_model_decoder, N_d=hp.n_layer_decoder, n_head_decoder=hp.n_head_decoder,
                        ff_conv_kernel_size_decoder=hp.ff_conv_kernel_size_decoder, concat_after_decoder=hp.concat_after_decoder,
                        reduction_rate=hp.reduction_rate, dropout=0.0, dropout_postnet=0.0, CTC_training=hp.CTC_training,
                        n_bins=hp.nbins, f0_min=hp.f0_min, f0_max=hp.f0_max, energy_min=hp.energy_min, energy_max=hp.energy_max,
                        pitch_pred=hp.pitch_pred, energy_pred=hp.energy_pred,
                        output_type=hp.output_type, num_group=hp.num_group, multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.num_speaker, spkr_emb=hp.spkr_emb)

    model.to(DEVICE)
    model.eval()

    model.load_state_dict(load_model(f"{load_name}"), strict=False)
    # To FastSpeech2 dataset
    dataset_test = datasets.DevDatasets(hp.test_script, hp, alignment_pred=True, pitch_pred=True, energy_pred=True)
    collate_fn_transformer = datasets.collate_fn
    sampler = datasets.NumBatchSampler(dataset_test, 1, shuffle=False)

    dataloader = DataLoader(dataset_test, batch_sampler=sampler, num_workers=1, collate_fn=collate_fn_transformer)

    if hp.mean_file is not None and hp.var_file is not None:
        mean_value = np.load(hp.mean_file).reshape(-1, hp.mel_dim)
        var_value = np.load(hp.var_file).reshape(-1, hp.mel_dim)

    import time
    all_time = time.time()
    for idx, d in enumerate(dataloader):
        text, mel, pos_text, pos_mel, text_lengths, mel_lengths, stop_token, spk_emb, f0, energy, alignment, mel_name = d

        text = text.to(DEVICE, non_blocking=True)
        mel_input = mel.to(DEVICE, non_blocking=True)
        pos_text = pos_text.to(DEVICE, non_blocking=True)
        pos_mel = pos_mel.to(DEVICE, non_blocking=True)
        mel_lengths = mel_lengths.to(DEVICE, non_blocking=True)
        text_lengths = text_lengths.to(DEVICE, non_blocking=True)
        stop_token = stop_token.to(DEVICE, non_blocking=True)
        if hp.is_multi_speaker:
            spk_emb = spk_emb.to(DEVICE, non_blocking=True)
        if hp.pitch_pred:
            f0 = f0.to(DEVICE, non_blocking=True)
        if hp.energy_pred:
            energy = energy.to(DEVICE, non_blocking=True)
        if hp.model.lower() == 'fastspeech2':
            alignment = alignment.to(DEVICE, non_blocking=True)

        src_mask, trg_mask = create_masks(pos_text, pos_mel, task=hp.model)
        with torch.no_grad():
            start_time = time.time()
            outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, attn_enc, attn_dec = model(text, src_mask, mel_mask=None, d_target=None, p_target=None, e_target=None, spkr_emb=spk_emb)
            if hp.CTC_training:
                outputs_prenet, outputs_postnet, outputs_stop_token, attn_enc, attn_dec_dec, attn_dec_enc, ctc_outputs, results_each_layer = model(text, mel_input, src_mask, trg_mask, spkr_emb=spk_emb)
            else:
                if hp.model.lower() == 'fastspeech2':
                    outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, attn_enc, attn_dec_dec = model(text, src_mask, trg_mask, alignment, f0, energy, spkr_emb=spk_emb)
                else:
                    outputs_prenet, outputs_postnet, outputs_stop_token, attn_enc, attn_dec_dec, attn_dec_enc, results_each_layer = model(text, mel_input, src_mask, trg_mask, spkr_emb=spk_emb)
        outputs = outputs_postnet[0].cpu().detach().numpy()
        if hp.var_file is not None:
            outputs *= np.sqrt(var_value)
        if hp.mean_file is not None:
            outputs += mean_value

        if hp.output_type == 'softmax':
            pred1 = outputs[:, :512].argmax(1)
            pred2 = outputs[:, 512:].argmax(1)
            outputs = np.vstack((pred1, pred2))

        base_name = mel_name[0]
        # base_name = os.path.splitext(os.path.basename(mel_name[0]))[0]
        # output_name = os.path.join(save_path, base_name + '_gen.npy')
        output_name = base_name.replace('.npy', '_gen.npy')
        print(f'save {output_name} {outputs.shape}')
        np.save(output_name, outputs)
        print(f'{time.time() - start_time}')
        sys.stdout.flush()
    print(f'elapsed time {time.time()-all_time}')
