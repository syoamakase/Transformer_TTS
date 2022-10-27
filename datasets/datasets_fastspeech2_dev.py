# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import librosa
import collections
import os
from typing import Optional
from operator import itemgetter
import random
from struct import unpack, pack
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
import sentencepiece as spm
from tqdm import tqdm

class DevDatasets(Dataset):
    """
    Train dataset.
    """                                                   
    def __init__(self, csv_file, hp, alignment_pred=True, pitch_pred=True, energy_pred=True, accent_emb=False):
        """
        Args:                                                                   
            csv_file (string): Path to the csv file with annotations.                
        """
        self.hp = hp
        self.landmarks_frame = pd.read_csv(csv_file, sep='\|', header=None)
        if self.hp.spm_model is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(self.hp.spm_model)

        if self.hp.mean_file is not None and self.hp.var_file is not None:
            self.mean_value = np.load(self.hp.mean_file).reshape(-1, self.hp.mel_dim)
            self.var_value = np.load(self.hp.var_file).reshape(-1, self.hp.mel_dim)

        self.pred_alignment = alignment_pred
        self.pred_f0 = pitch_pred
        self.pred_energy = energy_pred
        self.accent_emb = accent_emb

    def load_htk(self, filename):
        fh = open(filename, "rb")
        spam = fh.read(12)
        _, _, sampSize, _ = unpack(">IIHH", spam)
        veclen = int(sampSize / 4)
        fh.seek(12, 0)
        dat = np.fromfile(fh, dtype='float32')
        dat = dat.reshape(int(len(dat) / veclen), veclen)
        dat = dat.byteswap()
        fh.close()
        return dat

    def __len__(self):                                                          
        return len(self.landmarks_frame)

    def __getitem__(self, idx): 
        mel_name = self.landmarks_frame.loc[idx, 0]
        alignment_name = mel_name.replace('.npy', '_alignment.npy')
        f0_name = mel_name.replace('.npy', '_f0.npy')
        energy_name = mel_name.replace('.npy', '_energy.npy')
        text = self.landmarks_frame.loc[idx, 1].strip()
        
        if self.hp.is_multi_speaker:
            spk_emb_name = self.landmarks_frame.loc[idx, 2] #self.landmarks_frame.loc[idx, 2].strip()
            if self.hp.spk_emb_type == 'speaker_id':
                spk_emb = int(spk_emb_name)
            else:
                spk_emb = np.load(spk_emb_name.strip())
        else:
            spk_emb = None

        if self.hp.spm_model is not None:
            textids = [self.sp.bos_id()] + self.sp.EncodeAsIds(text)+ [self.sp.eos_id()]
            text = np.array([int(t) for t in textids], dtype=np.int32)
        else:
            text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)
        if '.npy' in mel_name:
            mel_input = np.load(mel_name)
            assert mel_input.shape[0] == self.hp.mel_dim or mel_input.shape[1] == self.hp.mel_dim, '{} does not have strange shape {}'.format(mel_name, mel_input.shape)
            if mel_input.shape[1] != self.hp.mel_dim:
                mel_input = mel_input.reshape(-1, self.hp.mel_dim)
        elif '.htk' in mel_name:
            mel_input = self.load_htk(mel_name)[:,:self.hp.mel_dim]
        elif '.mel' in mel_name:
            mel_input = torch.load(mel_name).squeeze(0).transpose(0,1).numpy()
        else:
            raise ValueError('{} is unknown file extension. Please check the extension or change htk or npy'.format(mel_name))

        if self.pred_alignment:
            alignment = np.load(alignment_name)
        else:
            alignment = None
        if self.pred_f0:
            f0 = np.load(f0_name)
        else:
            f0 = None
        if self.pred_energy:
            energy = np.load(energy_name)
        else:
            energy = None
        
        if self.accent_emb:
            accentids = self.landmarks_frame.loc[idx, 2]
            accent = np.array([int(t) for t in accentids.split(' ')], dtype=np.int32) 
        else:
            accent = None

        if self.hp.mean_file is not None and self.hp.var_file is not None:
            mel_input -= self.mean_value
            mel_input /= np.sqrt(self.var_value)

        if self.hp.model.lower() == 'fastspeech2':
            mel_length = mel_input.shape[0]
            reduction_rate = 1
        else:
            mel_input = np.concatenate([np.zeros([1, self.hp.mel_dim], np.float32), mel_input], axis=0)
            mel_length = _round_up(mel_input.shape[0], self.hp.reduction_rate)
            reduction_rate = self.hp.reduction_rate

        text_length = len(text)
        stop_token = torch.zeros(mel_input.shape[0])
        pos_text = np.arange(1, text_length+1)
        pos_mel = np.arange(1, mel_length+1)

        sample = {'text': text, 'text_length':text_length, 'mel_input':mel_input, 'mel_length':mel_length, 'pos_mel':pos_mel,
                  'pos_text':pos_text, 'stop_token':stop_token, 'spk_emb':spk_emb, 'f0':f0, 'energy':energy, 'alignment':alignment, 'accent':accent,
                  'reduction_rate': reduction_rate, 'is_multi_speaker':self.hp.is_multi_speaker, 'spk_emb_type': self.hp.spk_emb_type,
                  'mel_name':mel_name}
        return sample


def collate_fn(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        reduction_rate = batch[0]['reduction_rate']
        spk_emb_type = batch[0]['spk_emb_type']
        is_multi_speaker = batch[0]['is_multi_speaker']

        text = [d['text'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        mel_name = [d['mel_name'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        mel_length = [d['mel_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        stop_token = [d['stop_token'] for d in batch]
        spk_emb = [d['spk_emb'] for d in batch]
        f0 = [d['f0'] for d in batch]
        energy = [d['energy'] for d in batch]
        alignment = [d['alignment'] for d in batch]
        accent = [d['accent'] for d in batch]
        
        pred_alignment = alignment[0] is not None
        pred_f0 = f0[0] is not None
        pred_energy = energy[0] is not None
        accent_input = accent[0] is not None

        text = _prepare_data(text).astype(np.int32)
        mel_input = _pad_mel(mel_input, reduction_rate)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)
        if pred_f0:
            f0 = _prepare_data(f0).astype(np.float64)
            f0 = torch.FloatTensor(f0)
        else:
            f0 = None
        if pred_energy:
            energy = _prepare_data(energy).astype(np.float64)
            energy = torch.FloatTensor(energy)
        else:
            energy = None

        if pred_alignment:
            alignment = _prepare_data(alignment).astype(np.int32)
            alignment = torch.LongTensor(alignment)
        else:
            alignment = None

        if accent_input:
            accent = _prepare_data(accent).astype(np.int32)
            accent = torch.LongTensor(accent)
        else:
            accent = None
        stop_token = _pad_stop_token(stop_token, reduction_rate, _pad=1.0)

        text = torch.LongTensor(text)
        mel_input = torch.FloatTensor(mel_input)
        pos_text = torch.LongTensor(pos_text)
        pos_mel = torch.LongTensor(pos_mel)
        text_length = torch.LongTensor(text_length)
        mel_length = torch.LongTensor(mel_length)
        stop_token = torch.FloatTensor(stop_token)

        if is_multi_speaker:
            if spk_emb_type == 'x_vector':
                return text, mel_input, pos_text, pos_mel, text_length, mel_length, stop_token, torch.FloatTensor(spk_emb), f0, energy, alignment, mel_name, accent
            elif spk_emb_type == 'speaker_id':
                return text, mel_input, pos_text, pos_mel, text_length, mel_length, stop_token, torch.LongTensor(spk_emb), f0, energy, alignment, mel_name, accent
            else:
                raise AttributeError(f'{spk_emb_type} is unknown in spk_emb_type')
        else:
            return text, mel_input, pos_text, pos_mel, text_length, mel_length, stop_token, None, torch.FloatTensor(f0), energy, alignment, mel_name, accent

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                    .format(type(batch[0]))))


def _pad_data(x, length, pad_value=0):
    _pad = pad_value
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def _round_up(x, multiple):
    remainder = x % multiple
    #return x if remainder == 0 else x + multiple + remainder
    return x if remainder == 0 else x + multiple - remainder

def _pad_mel(inputs, reduction_rate=1, _pad=None):
    if not _pad:
        #if hp.mean_file is None and hp.var_file is None:
        #     _pad = -5.0
        #else:
        _pad = -0.5
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        max_len_reduction = _round_up(max_len, reduction_rate)
        return np.pad(x, [[0, max_len_reduction - mel_len], [0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

def _pad_stop_token(inputs, reduction_rate=1, _pad=1.0):
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        max_len_reduction = _round_up(max_len, reduction_rate)
        return np.pad(x, (0, max_len_reduction - mel_len), mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

class LengthsBatchSampler(Sampler):
    """
    LengthsBatchSampler - Sampler for variable batch size. Mainly, we use it for Transformer.
    It requires lengths.

    Args:
        dataset (torch.nn.dataset)
    """
    def __init__(self, dataset, n_lengths, hp, lengths_file=None, shuffle=True, shuffle_one_time=False, reverse=False):
        assert not ((shuffle == reverse) and shuffle is True), 'shuffle and reverse cannot set True at the same time.'

        if lengths_file is None or not os.path.exists(lengths_file):
            print('lengths_file is not exists. Make...')
            loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=1)
            lengths_list = []
            pbar = tqdm(loader)
            for d in pbar:
                mel_input = d[1]
                # print(mel_input.shape)
                lengths_list.append(mel_input.shape[1])
            self.lengths_np = np.array(lengths_list)
            np.save(hp.lengths_file, self.lengths_np)
        else:
            print('{} is loading.'.format(lengths_file))
            self.lengths_np = np.load(lengths_file)
            assert len(dataset) == len(self.lengths_np), 'mismatch the number of lines between dataset and {}'.format(lengths_file)
        
        self.n_lengths = n_lengths
        self.all_indices = self._batch_indices()
        if shuffle_one_time:
            random.shuffle(self.all_indices)
        self.shuffle = shuffle
        self.shuffle_one_time = shuffle_one_time
        self.reverse = reverse

    def _batch_indices(self):
        self.count = 0
        all_indices = []
        while self.count + 1 < len(self.lengths_np):
            indices = []
            max_len = 0
            while self.count < len(self.lengths_np):
                curr_len = self.lengths_np[self.count]
                mel_lengths = max(max_len, curr_len) * (len(indices) + 1)
                if mel_lengths > self.n_lengths or (self.count + 1) > len(self.lengths_np):
                    break
                max_len = max(max_len, curr_len)
                #mel_lengths += curr_len
                indices.extend([self.count])
                self.count += 1
            all_indices.append(indices)
       
        return all_indices

    def __iter__(self):
        if self.shuffle and not self.shuffle_one_time:
            random.shuffle(self.all_indices)
        if self.reverse:
            self.all_indices.reverse()

        for indices in self.all_indices:
            yield indices

    def __len__(self):
        return len(self.all_indices)

class NumBatchSampler(Sampler):
    """
    """
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True):
        self.batch_size = batch_size
        self.drop_last = drop_last 
        self.dataset_len = len(dataset)
        self.all_indices = self._batch_indices()
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.all_indices)

    def _batch_indices(self):
        batch_len = self.dataset_len // self.batch_size
        mod = self.dataset_len % self.batch_size
        all_indices = np.arange(self.dataset_len-mod).reshape(batch_len, self.batch_size).tolist()
        if mod != 0:
            remained = np.arange(self.dataset_len-mod, self.dataset_len).tolist()
            all_indices.append(remained) 
       
        return all_indices

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.all_indices)

        for indices in self.all_indices:
            yield indices

    def __len__(self):
        return len(self.all_indices)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

def get_dataset(script_file='examples/LJSpeech/data/train/script_16000/train_id_sort_xlen.txt'):
    print(f'script_file = {script_file}')
    return TrainDatasets(script_file)

def get_test_dataset(script_file='examples/LJSpeech/data/dev/script_16000/dev_id.txt'):
    print(f'script_file = {script_file}')
    return TestDatasets(script_file)

if __name__ == '__main__':
    from utils import hparams as hp
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py')
    parser.add_argument('--train_script', default=None)
    args = parser.parse_args()

    hp.configure(args.hp_file)
    if args.train_script is not None:
        hp.train_script = args.train_script
    print(f'train script = {hp.train_script}')
    datasets = TrainDatasets(hp.train_script, hp)
    sampler = LengthsBatchSampler(datasets, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False)
    dataloader = DataLoader(datasets, batch_sampler=sampler, num_workers=8, collate_fn=collate_fn)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        #text, mel_input, pos_text, pos_mel, text_lengths, mel_lengths = d
        text, mel, pos_text, pos_mel, text_lengths, mel_lengths, stop_token, spk_emb, f0, energy, alignment = d
        text = text.to(DEVICE, non_blocking=True)
        mel = mel.to(DEVICE, non_blocking=True)
        pos_text = pos_text.to(DEVICE, non_blocking=True)
        pos_mel = pos_mel.to(DEVICE, non_blocking=True)
        text_lengths = text_lengths.to(DEVICE, non_blocking=True)
