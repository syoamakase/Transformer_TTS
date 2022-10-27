# -*- coding: utf-8 -*-
import argparse
import copy
from collections import OrderedDict
import itertools
import os
from struct import unpack, pack
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import math
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm
import datasets.datasets_fastspeech2 as datasets
from torch.utils.data import DataLoader

from utils import hparams as hp
from utils.utils import log_config, fill_variables

#from Models.transformer import Transformer
from Models.fastspeech2 import FastSpeech2

from Models.postnets import PostLowEnergyv1, PostLowEnergyv2

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

def nopeak_mask(size):
    """
    npeak_mask(4)
    >> tensor([[[ 1,  0,  0,  0],
         [ 1,  1,  0,  0],
         [ 1,  1,  1,  0],
         [ 1,  1,  1,  1]]], dtype=torch.uint8)

    """
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
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
    parser.add_argument('--load_name_tts', default=None)
    parser.add_argument('--test_script', default=None)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    load_name_tts = args.load_name_tts
    load_name = args.load_name
    test_script = args.test_script

    if os.path.exists(os.path.join(os.path.dirname(load_name), 'hparams.py')):
        hp_file = os.path.join(os.path.dirname(load_name), 'hparams.py')

    hp.configure(hp_file)
    fill_variables(hp)
    epoch = os.path.basename(load_name).replace('network.average_', '')
    save_path = os.path.join(os.path.dirname(load_name), 'dev/'+epoch)
    os.makedirs(save_path, exist_ok=True)

    assert hp.architecture == 'mel-mel', f'invalid architecture {hp.architecture}'
    if test_script is not None:
        hp.test_script = test_script

    if load_name_tts is None:
        load_name_tts = hp.pretrain_model

    # import pdb; pdb.set_trace()
    # initialize pytorch
    pretrained_model = FastSpeech2(hp, src_vocab=hp.vocab_size, trg_vocab=hp.mel_dim, d_model_encoder=hp.d_model_encoder, N_e=hp.n_layer_encoder,
                        n_head_encoder=hp.n_head_encoder, ff_conv_kernel_size_encoder=hp.ff_conv_kernel_size_encoder,
                        concat_after_encoder=hp.concat_after_encoder,
                        d_model_decoder=hp.d_model_decoder, N_d=hp.n_layer_decoder, n_head_decoder=hp.n_head_decoder,
                        ff_conv_kernel_size_decoder=hp.ff_conv_kernel_size_decoder, concat_after_decoder=hp.concat_after_decoder,
                        reduction_rate=hp.reduction_rate, dropout=0.0, dropout_postnet=0.0, dropout_variance_adaptor=0.0,
                        n_bins=hp.nbins, f0_min=hp.f0_min, f0_max=hp.f0_max, energy_min=hp.energy_min, energy_max=hp.energy_max,
                        pitch_pred=hp.pitch_pred, energy_pred=hp.energy_pred, 
                        output_type=hp.output_type, num_group=hp.num_group, multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim, spk_emb_architecture=hp.spk_emb_architecture)


    pretrained_model.to(DEVICE)
    pretrained_model.eval()
    pretrained_model.load_state_dict(load_model(f"{load_name_tts}"))
    
    if hp.version == 1 or hp.version == 5:
        model = PostLowEnergyv1(vocab_size=hp.mel_dim, out_size=hp.mel_dim_post, d_model=hp.d_model_encoder, N=hp.n_layer_encoder,
                               heads=hp.n_head_encoder, ff_conv_kernel_size=hp.ff_conv_kernel_size_encoder,
                               concat_after_encoder=hp.concat_after_encoder, dropout=0.0,
                               multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.num_speaker)
    elif hp.version == 2 or hp.version == 3 or hp.version == 4 or hp.version == 6:
        print(f'version {hp.version}')
        spk_emb_dim = hp.spk_emb_dim_postprocess #if hp.spk_emb_postprocess_type is not None else None
        model = PostLowEnergyv2(hp, vocab_size=hp.mel_dim, out_size=hp.mel_dim_post, d_model=hp.d_model_encoder, N=hp.n_layer_encoder,
                                heads=hp.n_head_encoder, ff_conv_kernel_size=hp.ff_conv_kernel_size_encoder,
                                concat_after_encoder=hp.concat_after_post, dropout=0.0,
                                multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim_postprocess, gender_emb=hp.gender_emb, speaker_emb=hp.speaker_emb,
                                concat=hp.concat, spk_emb_postprocess_type=hp.spk_emb_postprocess_type)
                                #multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.num_speaker, spk_emb_layer=[1, 2])
            #model = PostLowEnergyv2(hp, vocab_size=hp.mel_dim, out_size=hp.mel_dim_post, d_model=hp.d_model_encoder, N=hp.n_layer_encoder,
            #                        heads=hp.n_head_encoder, ff_conv_kernel_size=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.concat_after_post, dropout=hp.dropout,
            #                        multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim_postprocess, gender_emb=hp.gender_emb, speaker_emb=hp.speaker_emb, concat=hp.concat,spk_emb_postprocess_type=hp.spk_emb_postprocess_type)

    print('pretrain model')
    print(pretrained_model)
    print('post process')
    print(model) 
    model.to(DEVICE)
    model.eval()
    model.load_state_dict(load_model(f"{load_name}"))
    # To FastSpeech2 dataset
    if hp.output_type:
        dataset_test = datasets.VQWav2vecTestDatasets(hp.test_script)
        collate_fn_transformer = datasets.collate_fn_vqwav2vec_test
    else:
        dataset_test = datasets.TestDatasets(hp.test_script, hp)
        collate_fn_transformer = datasets.collate_fn_test
    sampler = datasets.NumBatchSampler(dataset_test, 1, shuffle=False)#hp.batch_size)

    dataloader = DataLoader(dataset_test, batch_sampler=sampler, num_workers=1, collate_fn=collate_fn_transformer)

    if hp.mean_file is not None and hp.var_file is not None:
        mean_value = np.load(hp.mean_file).reshape(-1, hp.mel_dim)
        var_value = np.load(hp.var_file).reshape(-1, hp.mel_dim)

    start_time = time.time()
    from tqdm import tqdm
    total_time = 0
    #for idx, d in tqdm(enumerate(dataloader)):
    for idx, d in enumerate(dataloader):
        # torch.LongTensor(text), mel_output, torch.LongTensor(pos_text), torch.LongTensor(text_length), spk_emb
        text, mel, pos_text, text_lengths, spk_emb, accent, gender, xvector = d

        text = text.to(DEVICE)
        #mel = mel.to(DEVICE)
        pos_text = pos_text.to(DEVICE)
        #pos_mel = pos_mel.to(DEVICE)
        text_lengths = text_lengths.to(DEVICE)
        #stop_token = stop_token.to(DEVICE)
        if spk_emb is not None:
            spk_emb = spk_emb.to(DEVICE)

        if gender is not None:
            gender = gender.to(DEVICE)
        
        if xvector is not None:
            xvector = xvector.to(DEVICE)

        src_mask = (pos_text != 0).unsqueeze(-2)
        with torch.no_grad():
            local_time = time.time()
            outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, variance_adaptor_output, text_dur_predicted, attn_enc, attn_dec, _, _, _, _, _  = pretrained_model(text, src_mask, mel_mask=None, d_target=None, p_target=None, e_target=None, spkr_emb=spk_emb, fix_mask=hp.fix_mask)
            # outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, variance_adaptor_output, text_dur_predicted, attn_enc, attn_dec, None, None, None, sq_vae_loss, sq_vae_perplexity
            if hp.postnet_pred:
                outputs = outputs_postnet
            else:
                outputs = outputs_prenet
            trg_mask = (torch.ones(outputs.shape[0], outputs.shape[1]) != 0).unsqueeze(-2).to(DEVICE)

            if hp.version == 1 or hp.version == 5:
                post_outputs = model(outputs, trg_mask)
            elif hp.version == 2 or hp.version == 3:
                post_outputs, _, _  = model(outputs, trg_mask, variance_adaptor_output, spkr_emb=xvector, gender=gender)
            elif hp.version == 4 or hp.version == 6:
                post_outputs,_, _ = model(outputs, trg_mask, text_dur_predicted)

        outputs =  outputs[0].cpu().detach().numpy()
        if hp.version <= 2 or hp.version == 4:
            new_outputs = copy.deepcopy(outputs)
            new_outputs[:, 0:80] = post_outputs[0].cpu().detach().numpy()
        elif hp.version == 3 or hp.version == 5 or hp.version == 6:
            #print('residual')
            # all dimensions
            new_net_outputs = post_outputs[0].cpu().detach().numpy()
            new_outputs = new_net_outputs + outputs
            
            # 40:80
            #final_outputs = outputs_postnet.clone()
            #final_outputs[:,:,0:20] = post_outputs[:,:,:] + final_outputs[:,:,0:20]
            #new_outputs = final_outputs.cpu().detach().numpy()[0]
            #

        if hp.var_file is not None:
            outputs *= np.sqrt(var_value)
            new_outputs *= np.sqrt(var_value)
        if hp.mean_file is not None:
            outputs += mean_value
            new_outputs += mean_value

        total_time += (time.time() - local_time)
        base_name = mel[0]
        if args.save:
            output_name = base_name
        else:
            base_name = os.path.splitext(os.path.basename(mel[0]))[0]
            output_name = os.path.join(save_path, base_name+'.npy')
        #output_name = os.path.join(save_path, str(idx)+'.npy')
        print(f'save {output_name} {outputs.shape}')
        np.save(output_name, new_outputs)
        sys.stdout.flush()
    print(f'elapsed time = {time.time() - start_time}')
    print(f'total time = {total_time}')
