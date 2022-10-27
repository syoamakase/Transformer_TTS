# -*- coding: utf-8 -*-
import numpy as np
from struct import unpack
import os
import random
import torch
import torch.nn as nn

#from utils import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def freq_mask(spec, F=10, num_masks=1, replace_with_zero=False, random_mask=False, granularity=1):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]
    for i in range(0, num_masks):
        f = random.randrange(0, F, granularity)
        if random_mask:
            sample = np.arange(0, num_mel_channels)
            masks = random.sample(list(sample), f)
            if (replace_with_zero): cloned[:, masks] = 0
            else: cloned[:, masks] = cloned.mean()
        else:
            f_zero = random.randrange(0, num_mel_channels - f)
            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f*granularity): return cloned
            mask_end = random.randrange(f_zero, f_zero + f, granularity)
            if (replace_with_zero): cloned[:, f_zero:mask_end] = 0
            else: cloned[:, f_zero:mask_end] = cloned.mean()
    return cloned

def time_mask(spec, T=50, num_masks=1, replace_with_zero=False, random_mask=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[0]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero): cloned[t_zero:mask_end,:] = 0
        else: cloned[t_zero:mask_end,:] = cloned.mean()
    return cloned
                                                                                

def spec_augment(spec, T, F, num_T=1, num_F=1):
    # spec (B, T, F)
    for i in range(spec.shape[0]):
        spec[i] = time_mask(spec[i], T=T, num_masks=num_T, replace_with_zero=True)
        spec[i] = freq_mask(spec[i], F=F, num_masks=num_F, replace_with_zero=True)

    return spec
    
def log_config(hp):
    """To display the parameters of hparams.py
    """
    print(f'PID = {os.getpid()}')
    print(f'PyTorch version = {torch.__version__}')
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print('cuda device = {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    for key in hp.__dict__.keys():
        if not '__' in key:
            print('{} = {}'.format(key, eval('hp.'+key)))

def load_dat(filename):
    """To read binary data in htk file.
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

def onehot(labels, num_output):
    """
    To make onehot vector.
    ex) labels : 3 -> [0, 0, 1, 0, ...]

    Args:
        labels : true label ID
        num_output : the number of entry

    Return:
        utt_label : one hot vector.
    """
    utt_label = np.zeros((len(labels), num_output), dtype='float32')
    for i, label in enumerate(labels):
        utt_label[i][label] = 1.0
    return utt_label

def load_model(model_file, map_location=DEVICE):
    """
    To load the both of multi-gpu model and single gpu model.
    """
    model_state = torch.load(model_file, map_location=map_location)
    is_multi_loading = True if torch.cuda.device_count() > 1 else False
    # This line may include bugs!!
    is_multi_loaded = True if 'module' in list(model_state.keys())[0] else False

    return_model = None
    if is_multi_loaded is is_multi_loading:
        return_model = model_state

    elif is_multi_loaded is False and is_multi_loading is True:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state['module.'+key] = model_state[key]

        return_model = new_model_state

    elif is_multi_loaded is True and is_multi_loading is False:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]

        return_model = new_model_state

    return return_model

def adjust_learning_rate(optimizer, epoch, hp):
    """ Change learning rate to improve the performance in the attention model.
    
    Args:
        optimizer (torch.nn.optim): Optimizer which you want to change the learning rate.
        epoch (int): Epoch which the model finished.
    """
    if hp.reset_optimizer_epoch is not None:
        if (epoch % hp.reset_optimizer_epoch) > hp.lr_adjust_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
    else:
        assert len(hp.lr_adjust_epoch) == 1, 'you must set reset_optimizer_epoch when setting multiple lr_adjust_epoch'
        if epoch > hp.lr_adjust_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8

def init_weight(m):
    """To initialize weights and biases.
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

def overwrite_hparams(args):
    for key, value in args._get_kwargs():
        if value is not None and value != 'load_name':
            setattr(hp, key, value)

def fill_variables(hp):
    default_var = {'spm_model':None, 'mean_file':None, 'var_file': None, 'log_dir': 'logs',
                   'positive_weight':5.0, 'is_multi_speaker':False, 'num_speaker': None, 'spk_emb_type': None, 'spk_emb_architecture': '',
                    'output_type':None, 'num_group':None, 'pitch_pred':True, 'energy_pred':True, 'model':'Fastspeech2', 'amp':True, 'gst':False,
                    'encoder_type':'transformer', 'clip':1.0, 'decoder_type': 'transformer', 'accent_emb':False, 'channel_wise': False, 'tail_alignment':'_alignment',
                    'gender_emb':False, 'ctc_out':False, 'concat':False, 'vq_code':False, 'speaker_emb':False, 'spk_emb_postprocess_type': None, 'spk_emb_dim_postprocess':None, 'mask':False, 'post_conformer':False,
                    'fix_mask':None, 'use_cosine_emb_loss': False, 'n_layer_post_model':6, 'semantic_mask':False, 'time_weight': None, 'mask_probability':0.06, 'ff_conv_kernel_size_post':5, 'concat_after_post':True,
                    'intermediate_layers_out':None, 'dropout_variance_adaptor':0.5, 'use_sq_vae':False, 'spk_emb_dim':None, 'use_rnn_length':False, 'use_pos': False, 'p_scheduled_sampling':0.0, 'use_ssim':False,
                    'spk_emb_vers':1, 'decoder_type':'transformer', 'use_hop':False} 

    for key, value in default_var.items():
        if not hasattr(hp, key):
            print('{} is not found in hparams. defalut {} is used.'.format(key, value))
            setattr(hp, key, value)
    if hp.spk_emb_postprocess_type == 'x_vector' and hp.spk_emb_dim_postprocess is None:
        hp.spk_emb_dim_postprocess = 512

    assert not hasattr(hp, 'spkr_emb'), 'hp.spkr_emb is future depricated, please use hp.spk_emb_architecture instead.'


def get_learning_rate(step, d_model, warmup_factor, warmup_step):
    """To get a learning rate with Noam optimizer
    Args:
        step (int): The count of iteration in training
        d_model (int): The dimension of the hidden states
        warmup_factor (int): warmup factor
        warmup_step (int): warmup step

    Return:
        (float): The output learning rate
    """
    return warmup_factor * min(step ** -0.5, step * warmup_step ** -1.5) * (d_model ** -0.5)
