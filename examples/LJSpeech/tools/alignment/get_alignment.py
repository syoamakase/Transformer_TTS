# -*- coding: utf-8 
import argparse
import copy
import itertools
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import hparams as hp
from Models.CTCModel import CTCModel
from Models.CTCModel import CTCForcedAligner
import matplotlib.pyplot as plt
from utils.utils import log_config, load_model, overwrite_hparams, fill_variables
import datasets

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_loop(model):
    dataset_train = datasets.TestDatasets(hp.test_script, align=True)
    dataloader = DataLoader(dataset_train, batch_size=10, num_workers=4, collate_fn=datasets.collate_fn_align, drop_last=False)
    for d in dataloader:
        text, mel_input, pos_text, pos_mel, text_lengths, mel_lengths, mel_names = d

        text = text.to(DEVICE)
        mel_input = mel_input.to(DEVICE)
        pos_text = pos_text.to(DEVICE)
        pos_mel = pos_mel.to(DEVICE)
        text_lengths = text_lengths.to(DEVICE)

        if hp.frame_stacking > 1 and hp.encoder_type != 'Wave':
            mel_input, mel_lengths = frame_stacking(mel_input, mel_lengths, hp.frame_stacking)

        predict_ts = model.align(mel_input, mel_lengths, text)

        ctc_aligner = CTCForcedAligner(blank=0)
        trigger_points = ctc_aligner.align(predict_ts.cpu(), mel_lengths.cpu(), text.cpu(), text_lengths.cpu())

        # plt.imshow(mel_input[0].transpose(0,1).cpu().numpy())
        # for i in trigger_points[0]:
        #     plt.axvline(x=i, color='red')

        trigger_points += 1
        for mel_name, trigger in zip(mel_names, trigger_points):
            # print(f'{mel_name}', end='|')
            alignment_name = mel_name.replace('.npy', '_alignment.npy')
            alignment = []
            for i,t in enumerate(trigger):
                if i != 0 and int(t) == 1:
                    break
                if i == 0:
                    continue
                elif i == 1:
                    frames = int(t)
                    alignment.append(frames)
                    # print(f'{frames}', end=' ')
                else:
                    frames = int(t) - trigger[i-1]
                    alignment.append(frames)
                    # print(f'{frames}', end=' ')
            alignment = np.array(alignment)
            np.save(alignment_name, alignment)

        sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_name')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py')
    parser.add_argument('--test_script', type=str, default=None)
    
    args = parser.parse_args()
    load_name = args.load_name

    load_dir = os.path.dirname(load_name)
    if os.path.exists(os.path.join(load_dir, 'hparams.py')):
        args.hp_file = os.path.join(load_dir, 'hparams.py')

    hp.configure(args.hp_file)
    fill_variables()
    overwrite_hparams(args)
    
    assert hp.decoder_type == 'CTC', 'Alignment mode on attention model is not implemented, Sorry!'

    if hp.decoder_type == 'CTC':
        model = CTCModel()

    model = model.to(DEVICE)
    model.eval()

    model.load_state_dict(load_model(load_name))
    test_loop(model)
