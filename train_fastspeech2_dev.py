# -*- coding: utf-8 -*-
# v1: w/o the outputs of variance adaptor (replace)
# v2: w/ .. (replace)
# v3: residual w/ text
# v4: w/ the outputs of duration predictor (replace)
# v5: w/o the outputs of variance adaptor (residual)
# v6: w/ the outputs of duration predictor (residual)

import argparse
import copy
import filecmp
import os
import numpy as np
import random
import sys
import shutil
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from utils import hparams as hp
from utils.utils import log_config, fill_variables, init_weight, get_learning_rate, load_model, spec_augment

from Models.transformer import Transformer
from Models.fastspeech2 import FastSpeech2
## proposed
from Models.postnets import PostLowEnergyv1, PostLowEnergyv2
# from Models.lightspeech import LightSpeech
import datasets.datasets_fastspeech2 as datasets

port = '600021'

random.seed(77)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.cuda.amp.GradScaler()

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
    if task.lower() == 'fastspeech2' or task.lower() == 'lightspeech':
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

def loss_mel(hp, pred, y, channel_wise=False, loss='l1'):
    #post_mel_loss = nn.L1Loss()(outputs_postnet, mel[:,hp.reduction_rate:,:])
    if channel_wise:
        print('channel')
        loss = hp.channel_weight[0] * F.l1_loss(pred[:,:,:20], y[:, :, :20]) + hp.channel_weight[1] * F.l1_loss(pred[:,:,20:], y[:, :, 20:])
    else:
        loss = F.l1_loss(pred, y)
    
    return loss

def loss_mel_framewise(hp, pred, y):
    loss_all = torch.zeros((pred.shape[2]))
    for channel in range(pred.shape[2]):
        loss_all[channel] = F.l1_loss(pred[:,:,channel], y[:, :, channel])
    
    return loss_all

def train_loop(model, optimizer, pretrained_model, step, epoch, args, hp, rank, dataloader):
    loss_abc_sum = torch.zeros((80))
    loss_fastspeech2_sum = torch.zeros((80))
    batch_sum = 0
    iter_sum = 0
    for d in dataloader: 
        if hp.optimizer.lower() != 'radam':
            lr = get_learning_rate(step, hp.d_model_decoder, hp.warmup_factor, hp.warmup_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            text, mel, pos_text, pos_mel, text_lengths, mel_lengths, stop_token, spk_emb, f0, energy, alignment, accent, gender, spk_emb_postprocess, mel_name = d

            text = text.to(DEVICE, non_blocking=True)
            mel = mel.to(DEVICE, non_blocking=True)
            pos_text = pos_text.to(DEVICE, non_blocking=True)
            pos_mel = pos_mel.to(DEVICE, non_blocking=True)
            mel_lengths = mel_lengths.to(DEVICE, non_blocking=True)
            text_lengths = text_lengths.to(DEVICE, non_blocking=True)
            stop_token = stop_token.to(DEVICE, non_blocking=True)
            if hp.is_multi_speaker:
                spk_emb = spk_emb.to(DEVICE, non_blocking=True)

            if hp.spk_emb_postprocess_type:
                spk_emb_postprocess = spk_emb_postprocess.to(DEVICE, non_blocking=True)

            if hp.gender_emb:
                gender = gender.to(DEVICE, non_blocking=True)
            if hp.pitch_pred:
                f0 = f0.to(DEVICE, non_blocking=True)
            if hp.energy_pred:
                energy = energy.to(DEVICE, non_blocking=True)
            if hp.model.lower() == 'fastspeech2' or hp.model.lower() == 'lightspeech':
                alignment = alignment.to(DEVICE, non_blocking=True)

        batch_size = mel.shape[0]
        if hp.reduction_rate > 1:
            mel_input = mel[:,:-hp.reduction_rate:hp.reduction_rate,:]
            pos_mel = pos_mel[:,:-hp.reduction_rate:hp.reduction_rate]
            mel_lengths_reduction = (mel_lengths - hp.reduction_rate) // hp.reduction_rate
        elif hp.model.lower() != 'fastspeech2' and hp.model.lower() != 'lightspeech':
            mel_input = mel[:,:-1,:]
            pos_mel = pos_mel[:,:-1]

        src_mask, trg_mask = create_masks(pos_text, pos_mel, task=hp.model)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(hp.amp): #and torch.autograd.set_detect_anomaly(True):
            with torch.no_grad():
                #if hp.CTC_training:
                #    outputs_prenet, outputs_postnet, outputs_stop_token, variance_adaptor_output, attn_enc, attn_dec_dec, attn_dec_enc, ctc_outputs, results_each_layer = pretrained_model(text, mel_input, src_mask, trg_mask, spkr_emb=spk_emb)
                #else:
                if hp.model.lower() == 'fastspeech2':
                    outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, variance_adaptor_output, text_dur_predicted, attn_enc, attn_dec_dec = pretrained_model(text, src_mask, trg_mask, alignment, f0, energy, spkr_emb=spk_emb)
                else:
                    raise AttributeError

            ## Mask
            if hp.postnet_pred:
                input_meltomel = outputs_postnet
            else:
                input_meltomel = outputs_prenet

            if hp.semantic_mask:
                if hp.semantic_mask_phone:
                    mask_phone_feature = variance_adaptor_output
                else:
                    mask_phone_feature = None
                input_meltomel, mask_phone_feature, mask_frames = pretrained_model._semantic_mask(input_meltomel, alignment, mask_phone_feature, p=hp.mask_probability)
                phone_feature = variance_adaptor_output if mask_phone_feature is None else mask_phone_feature
            else:
                phone_feature = variance_adaptor_output
                mask_frames = None

            #if hp.mask:
            #    input_postprocess = spec_augment(input_postprocess, T=50, F=20, num_T=2, num_F=2)

            ## train new model for low-energy parts
            if hp.version == 1 or hp.version == 5:
                outputs = model(input_meltomel, trg_mask)
            elif hp.version == 2 or hp.version == 3 or hp.version == 7:
                outputs, ctc_outputs, diff = model(input_meltomel, trg_mask, phone_feature, spkr_emb=spk_emb_postprocess, gender=gender)
                #outputs = model(outputs_postnet, trg_mask, variance_adaptor_output)
            elif hp.version == 4 or hp.version == 6:
                print('text_dur')
                outputs, ctc_outputs, diff = model(input_meltomel, trg_mask, text_dur_predicted)

            loss = 0.0
            if hp.model.lower() == 'fastspeech2':
                fastspeech2_loss = loss_mel(hp, outputs_prenet, mel, channel_wise=False).item() #nn.L1Loss()(outputs_prenet, mel)
                if hp.postnet_pred:
                    res_mel = outputs_postnet
                else:
                    res_mel = outputs_prenet
                if hp.version == 3 or hp.version == 5 or hp.version == 6:
                    print('residual')
                    # all part
                    outputs = outputs + res_mel 
                    #outputs = outputs + torch.log(F.hardtanh(outputs_postnet, min_val=0.01, max_val=2.0))
                    loss = loss_mel(hp, outputs, mel[:,:,0:80], channel_wise=hp.channel_wise)
                    fastspeech2_loss = loss_mel(hp, outputs_prenet, mel, channel_wise=False).item() #nn.L1Loss()(outputs_prenet, mel)
                
                    #abc_loss = loss_mel_framewise(hp, outputs,mel)
                    #abc_fastspeech2_loss = loss_mel_framewise(hp, outputs_postnet, mel)
                    #loss_abc_sum += abc_loss
                    #loss_fastspeech2_sum += abc_fastspeech2_loss
                    #batch_sum += batch_size
                    #iter_sum += 1
                    #if batch_sum > 1000:
                    #    import pdb; pdb.set_trace()
                    ## part
                    #final_outputs = outputs_postnet.clone()
                    #final_outputs[:,:,0:60] = outputs[:,:,:] + final_outputs[:,:,0:60]
                    #import pdb;pdb.set_trace()
                    #loss = loss_mel(hp, final_outputs[:,:,0:60], mel[:,:,0:60], channel_wise=False)
                #elif hp.version == 7:
                #    print('mask')
                #    # all part
                #    outputs = outputs * F.hardtanh(outputs_postnet, min_val=0, max_val=2.0)
                #    loss = loss_mel(hp, outputs, mel[:,:,0:80], channel_wise=hp.channel_wise)
                else:
                    loss = loss_mel(hp, outputs, mel[:,:,0:80], channel_wise=hp.channel_wise)
                    #loss = loss_mel(hp, outputs, mel[:,:,0:80], channel_wise=False)
                #if hp.ctc_out:
                #    ctc_outputs = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                #    loss_ctc = F.ctc_loss(ctc_outputs, text, mel_lengths, text_lengths, blank=0)
                #    loss += 0.2 * loss_ctc
                #    print(f'loss_ctc = {loss_ctc.item()}')

                if hp.vq_code:
                    loss_vq = diff
                    print(f'loss_vq = {loss_vq.item()}')
                    loss += loss_vq
            
            print(f'fastspeech2_loss = {fastspeech2_loss}')
            print(f'loss_total = {loss.item()}')
            print(f'batch size = {batch_size}')
            print(f'lr = {lr}')
            print(f'step {step} / {len(dataloader)}')
            step += 1
            sys.stdout.flush()

            if not torch.isnan(loss):
                if hp.amp:
                    scaler.scale(loss).backward()
                    if hp.clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # backward
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            #    print('loss is nan')
            #    sys.exit(1)
    if rank == 0 and (epoch+1) >= (hp.max_epoch-10):
        torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
    elif rank == 0 and ((epoch+1) % hp.save_per_epoch >= (hp.save_per_epoch-10) or ((epoch+1) % hp.save_per_epoch == 0)):
        torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
    if rank == 0 and (epoch+1) % hp.save_per_epoch == 0:
        torch.save(optimizer.state_dict(), hp.save_dir+"/network.optimizer.epoch{}".format(epoch+1))
    return step

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def train_epoch(model, optimizer, pretrained_model, step, start_epoch, args, hp, rank):
    if hp.output_type == 'softmax':
        dataset_train = datasets.VQWav2vecTrainDatasets(hp, hp.train_script)
        collate_fn = datasets.collate_fn_vqwav2vec
    else:
        alignment_pred = (hp.model.lower() == 'fastspeech2' or hp.model.lower() == 'lightspeech')
        dataset_train = datasets.TrainDatasets(hp.train_script, hp, alignment_pred=alignment_pred, pitch_pred=hp.pitch_pred, energy_pred=hp.energy_pred)
        collate_fn = datasets.collate_fn
    if hp.batch_size is not None:
        sampler = datasets.NumBatchSampler(dataset_train, hp.batch_size)
    elif hp.max_seqlen is not None:
        sampler = datasets.LengthsBatchSampler(dataset_train, hp.max_seqlen, hp, hp.lengths_file,shuffle=True, shuffle_one_time=False)

    train_sampler = datasets.DistributedSamplerWrapper(sampler) if args.n_gpus > 1 else sampler
    dataloader = DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=8, collate_fn=collate_fn)

    for epoch in range(start_epoch, hp.max_epoch):
        start_time = time.time()
        step = train_loop(model, optimizer, pretrained_model, step, epoch, args, hp, rank, dataloader)
        print("EPOCH {} end".format(epoch+1))
        print(f'elapsed time {time.time() - start_time}')

def init_distributed(rank, n_gpus, port):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."

    torch.cuda.set_device(rank % n_gpus)

    os.environ['MASTER_ADDR'] = 'localhost' #dist_config.MASTER_ADDR
    os.environ['MASTER_PORT'] = port #dist_config.MASTER_PORT

    torch.distributed.init_process_group(
        backend='nccl', world_size=n_gpus, rank=rank
    )

def cleanup():
    torch.distributed.destroy_process_group()


def run_distributed(fn, args, hp):
    port = '60' + str(int(time.time()))[-4:]
    print(f'port = {port}')
    try:
        mp.spawn(fn, args=(args, hp, port), nprocs=args.n_gpus, join=True)
    except:
        cleanup()

def run_training(rank, args, hp, port=None):
    if args.n_gpus > 1:
        init_distributed(rank, args.n_gpus, port)
        torch.cuda.set_device(f'cuda:{rank}')

    if hp.model.lower() == 'fastspeech2':
        pretrained_model = FastSpeech2(hp=hp, src_vocab=hp.vocab_size, trg_vocab=hp.mel_dim, d_model_encoder=hp.d_model_encoder, N_e=hp.n_layer_encoder,
                            n_head_encoder=hp.n_head_encoder, ff_conv_kernel_size_encoder=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.concat_after_encoder,
                            d_model_decoder=hp.d_model_decoder, N_d=hp.n_layer_decoder, n_head_decoder=hp.n_head_decoder,
                            ff_conv_kernel_size_decoder=hp.ff_conv_kernel_size_decoder, concat_after_decoder=hp.concat_after_decoder,
                            reduction_rate=hp.reduction_rate, dropout=hp.dropout, dropout_postnet=0.5, 
                            n_bins=hp.nbins, f0_min=hp.f0_min, f0_max=hp.f0_max, energy_min=hp.energy_min, energy_max=hp.energy_max, pitch_pred=hp.pitch_pred, energy_pred=hp.energy_pred,
                            accent_emb=hp.accent_emb, output_type=hp.output_type, num_group=hp.num_group, multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim, spk_emb_architecture=hp.spk_emb_architecture)

        if hp.version == 1 or hp.version == 5:
            print(f'version {hp.version}')
            model = PostLowEnergyv1(vocab_size=hp.mel_dim, out_size=hp.mel_dim_post, d_model=hp.d_model_encoder, N=hp.n_layer_encoder,
                                    heads=hp.n_head_encoder, ff_conv_kernel_size=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.concat_after_post, dropout=hp.dropout,
                                    multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.num_speaker)
        elif hp.version == 2 or hp.version == 3 or hp.version == 4 or hp.version == 6 or hp.version == 7:
            print(f'version {hp.version}')
            model = PostLowEnergyv2(hp, vocab_size=hp.mel_dim, out_size=hp.mel_dim_post, d_model=hp.d_model_encoder, N=hp.n_layer_encoder,
                                    heads=hp.n_head_encoder, ff_conv_kernel_size=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.concat_after_post, dropout=hp.dropout,
                                    multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim_postprocess, gender_emb=hp.gender_emb, speaker_emb=hp.speaker_emb, concat=hp.concat,spk_emb_postprocess_type=hp.spk_emb_postprocess_type)
                                    #multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.num_speaker, spk_emb_layer=[1, 2, 3, 4])

    print('pretrained model')
    print(pretrained_model)
    print('post process model')
    print(model)
    model.apply(init_weight)
    model.train()

    if hp.pretrain_model is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        loaded_dict = load_model(hp.pretrain_model, map_location=map_location)
        pretrained_model.load_state_dict(loaded_dict, strict=True)
        print(f'{hp.pretrain_model} is loaded')
        del loaded_dict
        torch.cuda.empty_cache()
    
    max_lr = 1e-3
    if hp.optimizer.lower() == 'radam':
        import radam
        optimizer = radam.RAdam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)

    model = model.to(rank)
    pretrained_model.train()
    pretrained_model = pretrained_model.to(rank)
    if args.n_gpus > 1:
        model = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[rank])
    
    if args.n_gpus > 1:
        dist.barrier()
        # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    
    if hp.loaded_epoch is not None:
        start_epoch = hp.loaded_epoch
        load_dir = hp.loaded_dir
        print('epoch {} loaded'.format(hp.loaded_epoch))
        loaded_dict = load_model("{}".format(os.path.join(load_dir, 'network.epoch{}'.format(hp.loaded_epoch))), map_location=map_location)
        model.load_state_dict(loaded_dict)
        #if hp.is_flat_start:
        #    step = 1
        #    start_epoch = 0
        #    print('flat_start')
        #else:
        loaded_dict = torch.load("{}".format(os.path.join(load_dir, 'network.optimizer.epoch{}'.format(hp.loaded_epoch))), map_location=map_location)
        optimizer.load_state_dict(loaded_dict)
        step = loaded_dict['state'][0]['step'] 
        del loaded_dict
        torch.cuda.empty_cache()
    else:
        start_epoch = 0
        step = 1
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('params = {0:.2f}M'.format(pytorch_total_params / 1000 / 1000))
    # train_loop(model, optimizer, step, epoch, args, hp, rank)
    train_epoch(model, optimizer, pretrained_model, step, start_epoch, args, hp, rank)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', type=str, default='hparams.py')
    args = parser.parse_args()
    hp_file = args.hp_file

    hp.configure(hp_file)
    fill_variables(hp)
    log_config(hp)

    assert hp.architecture == 'mel-mel', f'invalid architecture {hp.architecture}'
    save_dir = hp.save_dir # save dir name
    os.makedirs(save_dir, exist_ok=True)
    if hp_file != f'{save_dir}/hparams.py':
        if os.path.exists(f'{save_dir}/hparams.py'):
            if not filecmp.cmp(hp_file, f'{save_dir}/hparams.py'):
                shutil.copyfile(hp_file, f'{save_dir}/hparams.py')
        else:
            shutil.copyfile(hp_file, f'{save_dir}/hparams.py')
    #writer = SummaryWriter(f'{hp.log_dir}/logs/{hp.comment}')

    n_gpus = torch.cuda.device_count()
    args.__setattr__('n_gpus', n_gpus)

    if n_gpus > 1:
        run_distributed(run_training, args, hp)
    else:
        run_training(0, args, hp, None)
