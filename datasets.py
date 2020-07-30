# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import librosa
import collections
import os
import random
from struct import unpack, pack
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import librosa
import sentencepiece as spm
from tqdm import tqdm 

from utils import hparams as hp

class TrainDatasets(Dataset):                                                     
    """
    Train dataset.
    """                                                   
    def __init__(self, csv_file):                                     
        """                                                                     
        Args:                                                                   
            csv_file (string): Path to the csv file with annotations.           
            root_dir (string): Directory with all the wavs.                     
        """                                                                     
        # self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)  
        self.landmarks_frame = pd.read_csv(csv_file, sep='\|', header=None)    
        if hp.spm_model is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(hp.spm_model)

        if hp.mean_file is not None and hp.var_file is not None:
            self.mean_value = np.load(hp.mean_file).reshape(-1, hp.mel_dim)
            self.var_value = np.load(hp.var_file).reshape(-1, hp.mel_dim)
                                                                                
    def load_wav(self, filename):                                               
        return librosa.load(filename, sr=hp.sample_rate) 

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
        text = self.landmarks_frame.loc[idx, 1].strip()
        if hp.is_multi_speaker:
            spk_emb_name = self.landmarks_frame.loc[idx, 2].strip()
            spk_emb = np.load(spk_emb_name)
        else:
            spk_emb = None

        if hp.spm_model is not None:
            textids = [self.sp.bos_id()] + self.sp.EncodeAsIds(text)+ [self.sp.eos_id()]
            text = np.array([int(t) for t in textids], dtype=np.int32)
        else:
            text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)
        if '.npy' in mel_name:
            mel_input = np.load(mel_name)
            assert mel_input.shape[0] == hp.mel_dim or mel_input.shape[1] == hp.mel_dim, '{} does not have strange shape {}'.format(mel_name, mel_input.shape)
            if mel_input.shape[1] != hp.mel_dim:
                mel_input = mel_input.reshape(-1, hp.mel_dim)
        elif '.htk' in mel_name:
            mel_input = self.load_htk(mel_name)[:,:hp.mel_dim]
        elif '.mel' in mel_name:
            mel_input = torch.load(mel_name).squeeze(0).transpose(0,1).numpy()
        else:
            raise ValueError('{} is unknown file extension. Please check the extension or change htk or npy'.format(mel_name))
        
        if hp.mean_file is not None and hp.var_file is not None:
            mel_input -= self.mean_value
            mel_input /= np.sqrt(self.var_value)

        mel_input = np.concatenate([np.zeros([1,hp.mel_dim], np.float32), mel_input], axis=0)
        text_length = len(text)
        mel_length = _round_up(mel_input.shape[0], hp.reduction_rate)                           
        pos_text = np.arange(1, text_length+1)
        pos_mel = np.arange(1, mel_length+1)
        stop_token = torch.zeros(mel_length)

        sample = {'text': text, 'text_length':text_length, 'mel_input':mel_input, 'mel_length':mel_length, 'pos_mel':pos_mel, 'pos_text':pos_text, 'stop_token':stop_token, 'spk_emb':spk_emb}
                                                                                
        return sample

class TestDatasets(Dataset):                                                     
    """
    Test dataset.
    """                                                   
    def __init__(self, csv_file):                                     
        """                                                                     
        Args:                                                                   
            csv_file (string): Path to the csv file with annotations.           
            root_dir (string): Directory with all the wavs.                     
        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='\|', header=None)  
        #self.landmarks_frame = self._check_files()
        if hp.spm_model is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(hp.spm_model)

        if hp.mean_file is not None and hp.var_file is not None:
            self.mean_value = np.load(hp.mean_file).reshape(-1, hp.mel_dim)
            self.var_value = np.load(hp.var_file).reshape(-1, hp.mel_dim)           
                      
    def __len__(self):                                                          
        return len(self.landmarks_frame)                          

    def __getitem__(self, idx): 
        mel_output = self.landmarks_frame.loc[idx, 0]
        text = self.landmarks_frame.loc[idx, 1].strip()
        if hp.is_multi_speaker:
            spk_emb_name = self.landmarks_frame.loc[idx, 2].strip()
            spk_emb = np.load(spk_emb_name)
        else:
            spk_emb = None

        if hp.spm_model is not None:
            textids = [self.sp.bos_id()] + self.sp.EncodeAsIds(text)+ [self.sp.eos_id()]
            text = np.array([int(t) for t in textids], dtype=np.int32)
        else:
            text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)

        text_length = len(text)                        
        pos_text = np.arange(1, text_length+1)

        sample = {'text': text, 'text_length':text_length, 'mel_output':mel_output, 'pos_text':pos_text, 'spk_emb':spk_emb}
                                                                                
        return sample
    
    def _check_files(self):
        drop_indices = []
        for idx, mel_name in enumerate(self.landmarks_frame.loc[:,0]):
            if os.path.exists(mel_name):
                drop_indices.extend([idx])
        return self.landmarks_frame.drop(drop_indices).reset_index(drop=True)
    
def collate_fn_test(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        mel_output = [d['mel_output'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_text= [d['pos_text'] for d in batch]
        spk_emb = [d['spk_emb'] for d in batch]
        
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel_output = [i for i, _ in sorted(zip(mel_output, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        text_length = [i for i, _ in sorted(zip(text_length, text_length), key=lambda x: x[1], reverse=True)]
        spk_emb = [i for i,_ in sorted(zip(spk_emb, text_length), key=lambda x: x[1], reverse=True)]
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), mel_output, torch.LongTensor(pos_text), torch.LongTensor(text_length), None

def collate_fn(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        mel_length = [d['mel_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text= [d['pos_text'] for d in batch]
        stop_token = [d['stop_token'] for d in batch]
        spk_emb = [d['spk_emb'] for d in batch]
        
        text = [i for i,_ in sorted(zip(text, mel_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, mel_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, mel_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, mel_length), key=lambda x: x[1], reverse=True)]
        text_length =[i for i, _ in sorted(zip(text_length, mel_length), key=lambda x: x[1], reverse=True)]
        spk_emb = [i for i,_ in sorted(zip(spk_emb, mel_length), key=lambda x: x[1], reverse=True)]
        mel_length = sorted(mel_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)
        stop_token = torch.nn.utils.rnn.pad_sequence(stop_token, batch_first=True, padding_value=1)

        if hp.is_multi_speaker:
            if hp.spk_emb_type == 'x_vector':
                return torch.LongTensor(text), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length), torch.LongTensor(mel_length), torch.FloatTensor(stop_token), torch.LongTensor(spk_emb)
            elif hp.spk_emb_type == 'speaker_id':
                return torch.LongTensor(text), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length), torch.LongTensor(mel_length), torch.FloatTensor(stop_token), torch.LongTensor(spk_emb)
            else:
                raise AttributeError(f'{hp.spk_emb_type} is unknown in hp.spk_emb_type')
        else:
            return torch.LongTensor(text), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length), torch.LongTensor(mel_length), torch.FloatTensor(stop_token), None

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
    return x if remainder == 0 else x + multiple - remainder

def _pad_mel(inputs):
    if hp.mean_file is None and hp.var_file is None:
        _pad = -5.0
    else:
        _pad = -0.5
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        max_len_reduction = _round_up(max_len, hp.reduction_rate)
        return np.pad(x, [[0, max_len_reduction - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

class LengthsBatchSampler(Sampler):
    """
    LengthsBatchSampler - Sampler for variable batch size. Mainly, we use it for Transformer.
    It requires lengths.
    """
    def __init__(self, dataset, n_lengths, lengths_file=None, shuffle=True, shuffle_one_time=False, reverse=False):
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
            mel_lengths = 0
            while self.count < len(self.lengths_np):
                curr_len = self.lengths_np[self.count]
                if mel_lengths + curr_len > self.n_lengths or (self.count + 1) > len(self.lengths_np):
                    break
                mel_lengths += curr_len
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

def get_dataset(script_file='examples/LJSpeech/data/train/script_16000/train_id_sort_xlen.txt'):
    print(f'script_file = {script_file}')
    return TrainDatasets(script_file)

def get_test_dataset(script_file='examples/LJSpeech/data/dev/script_16000/dev_id.txt'):
    print(f'script_file = {script_file}')
    return TestDatasets(script_file)

if __name__ == '__main__':
    hp.configure('configs/hparams_LJSpeech.py')
    datasets = get_test_dataset('examples/LJSpeech/data/dev/script_16000/dev_id.txt')

    sampler = NumBatchSampler(datasets, 1, shuffle=False)
    dataloader = DataLoader(datasets, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn_test)

    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        print(d[1])
