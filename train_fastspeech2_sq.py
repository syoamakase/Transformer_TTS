# -*- coding: utf-8 -*-
import argparse
import copy
import filecmp
import os
import numpy as np
import random
import math
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
from utils.utils import log_config, fill_variables, init_weight, get_learning_rate, load_model

from Models.transformer import Transformer
from Models.fastspeech2_sq import SQFastSpeech2
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

# TODO: consider fixed lengths mask
def create_masks(src_pos, trg_pos, task='transformer', src_pad=0, trg_pad=0, debug=False):
    if debug:
        context_len = 7 # -3 +3
        src_mask = (src_pos != src_pad).unsqueeze(-2)
        size = src_pos.size(1)
        np_mask = np.zeros((1, size, size))#np.eye(size, k=0).astype('uint8')
        for k in range(-(context_len-1)//2,(context_len-1)//2+1):
            np_mask[0] += np.eye(size, k=k)

        np_mask = torch.from_numpy(np_mask.astype('uint8') == 1).to(DEVICE)
        src_mask_fixlen = src_mask & np_mask
    else:
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


def mse_loss_arelbo(input, target):
    # "Preventing Posterior Collapse Induced by Oversmoothing in Gaussian VAE"
    # https://arxiv.org/abs/2102.08663
    return 0.5 * (target.numel() // target.size(0)) * torch.log(torch.mean((input - target) ** 2))

def loss_mel(hp, pred, y, channel_wise=False, loss='l1', channel_weight=None):
    #post_mel_loss = nn.L1Loss()(outputs_postnet, mel[:,hp.reduction_rate:,:])
    if channel_wise:
        print('channel')
        loss = hp.channel_weight[0] * F.l1_loss(pred[:,:,:20], y[:, :, :20]) + hp.channel_weight[1] * F.l1_loss(pred[:,:,20:], y[:, :, 20:])
    else:
        loss = F.l1_loss(pred, y)
    
    return loss

def train_loop(model, optimizer, step, epoch, args, hp, rank, dataloader):
    channel_weight = torch.ones((hp.mel_dim))
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
            if hp.pitch_pred:
                f0 = f0.to(DEVICE, non_blocking=True)
            if hp.energy_pred:
                energy = energy.to(DEVICE, non_blocking=True)
            if hp.accent_emb:
                accent = accent.to(DEVICE, non_blocking=True)
            if hp.model.lower() == 'fastspeech2' or hp.model.lower() == 'lightspeech':
                alignment = None #alignment.to(DEVICE, non_blocking=True)

        batch_size = mel.shape[0]
        if hp.reduction_rate > 1:
            mel_input = mel[:,:-hp.reduction_rate:hp.reduction_rate,:]
            pos_mel = pos_mel[:,:-hp.reduction_rate:hp.reduction_rate]
            mel_lengths_reduction = (mel_lengths - hp.reduction_rate) // hp.reduction_rate
        elif hp.model.lower() != 'fastspeech2' and hp.model.lower() != 'lightspeech':
            mel_input = mel[:,:-1,:]
            pos_mel = pos_mel[:,:-1]

        src_mask, trg_mask = create_masks(pos_text, pos_mel, task=hp.model, debug=False)
        optimizer.zero_grad()

        if hp.use_sq_vae:
            # temperature = cfg.training.temperature.init * math.exp(-cfg.training.temperature.decay * global_step)
            temperature = 1.0 * math.exp(-0.00001 * step)
        else:
            temperature = None

        with torch.cuda.amp.autocast(hp.amp): #and torch.autograd.set_detect_anomaly(True):
            if False: #hp.CTC_training:
                outputs_prenet, outputs_postnet, outputs_stop_token, variance_adaptor_output, attn_enc, attn_dec_dec, attn_dec_enc, ctc_outputs, results_each_layer = model(text, mel_input, src_mask, trg_mask, spkr_emb=spk_emb)
            else:
                if hp.model.lower() == 'fastspeech2':
                    outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, variance_adaptor_output, text_dur_predicted, attn_enc, attn_dec_dec, _, _, _, sq_vae_loss, sq_vae_perplexity = model(text, src_mask, trg_mask, alignment, f0, energy, accent, spkr_emb=spk_emb, fix_mask=hp.fix_mask, temperature=temperature)
                elif hp.model.lower() == 'lightspeech':
                    outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, variance_adaptor_output, attn_enc, attn_dec_dec = model(text, src_mask, trg_mask, alignment, f0, energy, spkr_emb=spk_emb, ref_mel=mel, ref_mask=trg_mask)
                else:
                    raise AttributeError

            #attn_dec_dec = attn_dec_dec.cpu().numpy()
            #attn_enc = attn_enc.cpu().numpy()
            if hp.reduction_rate > 1:
                b,t,c = outputs_prenet.shape
                outputs_prenet = outputs_prenet.reshape(b, t*hp.reduction_rate, int(c//hp.reduction_rate))
                outputs_postnet = outputs_postnet.reshape(b, t*hp.reduction_rate, int(c//hp.reduction_rate))
                outputs_stop_token = outputs_stop_token.reshape(b, t*hp.reduction_rate)

            loss = 0.0
            if hp.output_type == 'softmax':
                mel1_loss = F.cross_entropy(outputs_prenet[:,:,:hp.mel_dim].transpose(1,2), mel[:,:,0], ignore_index=320)
                mel2_loss = F.cross_entropy(outputs_prenet[:,:,hp.mel_dim:].transpose(1,2), mel[:,:,1], ignore_index=320)
                post_mel1_loss = F.cross_entropy(outputs_postnet[:,:,:hp.mel_dim].transpose(1,2), mel[:,:,0], ignore_index=320)
                post_mel2_loss = F.cross_entropy(outputs_postnet[:,:,hp.mel_dim:].transpose(1,2), mel[:,:,1], ignore_index=320)
                mel_loss = mel1_loss + mel2_loss
                post_mel_loss = post_mel1_loss + post_mel2_loss
                acc1 = mel[:,:,0].eq(outputs_postnet[:,:,:hp.mel_dim].argmax(2)).sum() / (mel_lengths.sum() * hp.reduction_rate).float()
                acc2 = mel[:,:,1].eq(outputs_postnet[:,:,hp.mel_dim:].argmax(2)).sum() / (mel_lengths.sum() * hp.reduction_rate).float()
                print(f'accuracy_1 = {acc1}')
                print(f'accuracy_2 = {acc2}')
                #writer.add_scalar("Acc/accuracy_1", acc1, step)
                #writer.add_scalar("Acc/accuracy_2", acc2, step)
                loss = mel_loss + post_mel_loss
            else:
                if hp.model.lower() == 'fastspeech2':
                    if hp.channel_wise:
                        #mel_loss = nn.L1Loss()(outputs_prenet, mel) #loss_mel(hp, outputs_prenet, mel, channel_wise=True, channel_weight=channel_weight) #nn.L1Loss()(outputs_prenet, mel)
                        mel_loss = loss_mel(hp, outputs_prenet, mel, channel_wise=True, channel_weight=channel_weight) #nn.L1Loss()(outputs_prenet, mel)
                        if hp.postnet_pred:
                            post_mel_loss = loss_mel(hp, outputs_postnet, mel, channel_wise=True, channel_weight=channel_weight) #nn.L1Loss()(outputs_postnet, mel)
                        else:
                            post_mel_loss = 0.0
                        #post_mel_loss = nn.L1Loss()(outputs_postnet, mel)
                        loss = mel_loss + post_mel_loss #+ 0.3 * iter_loss
                    else:
                        if hp.use_sq_vae:
                            print('use_sq_vae')
                            mel_loss = mse_loss_arelbo(outputs_prenet, mel)
                        else:
                            mel_loss = nn.L1Loss()(outputs_prenet, mel)
                        if hp.postnet_pred:
                            post_mel_loss = nn.L1Loss()(outputs_postnet, mel)
                            loss = mel_loss + post_mel_loss #+ 0.3 * iter_loss
                        else:
                            loss = mel_loss

                print(f'loss_frame_before = {mel_loss.item()}')
            
            if hp.model.lower() == 'fastspeech2' or hp.model.lower() == 'lightspeech':
                # import pdb; pdb.set_trace()
                print(torch.exp(log_d_prediction))
                # import pdb; pdb.set_trace()
                duration_loss = 0
                for i_duration in range(batch_size):
                    duration_loss += nn.L1Loss()(torch.exp(log_d_prediction[i_duration, :text_lengths[i_duration]]).sum(), mel_lengths[i_duration]) # 1 = logoffset
                duration_loss /= batch_size
                #writer.add_scalar("Loss/train_duration", duration_loss, step)
                print(f'loss_duration = {duration_loss.item()}')
                if hp.pitch_pred:
                    f0_loss = nn.L1Loss()(p_prediction, f0)
                    loss += f0_loss
                    print(f'loss_f0 = {f0_loss.item()}')
                    #writer.add_scalar("Loss/train_f0", f0_loss, step)
                if hp.energy_pred:
                    energy_loss = nn.L1Loss()(e_prediction, energy)
                    loss += energy_loss
                    print(f'loss_energy = {energy_loss.item()}')
                    #writer.add_scalar("Loss/train_energy", energy_loss, step)
                
                loss += duration_loss
            else:
                loss_token = F.binary_cross_entropy_with_logits(outputs_stop_token, stop_token[:,hp.reduction_rate:], size_average=True, pos_weight=torch.tensor(hp.positive_weight))
                loss += loss_token
                print(f'loss_token = {loss_token.item()}')

            if hp.use_sq_vae:
                assert sq_vae_loss is not None and sq_vae_perplexity is not None
                loss += sq_vae_loss
                print(f"sq_vae_loss = {sq_vae_loss}")
                print(f"sq_vae_perplexity = {sq_vae_perplexity}")

            #print('lr = {}'.format(lr))
            if outputs_postnet is not None:
                print(f'loss_frame_after = {post_mel_loss.item()}')
            
            print(f'loss_total = {loss.item()}')
            print(f'batch size = {batch_size}')
            print(f'step {step} / {len(dataloader)}')
            assert not torch.isnan(loss), 'loss is nan'
            step += 1
            sys.stdout.flush()

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

def train_epoch(model, optimizer, step, start_epoch, args, hp, rank):
    if hp.output_type == 'softmax':
        dataset_train = datasets.VQWav2vecTrainDatasets(hp, hp.train_script)
        collate_fn = datasets.collate_fn_vqwav2vec
    else:
        alignment_pred = (hp.model.lower() == 'fastspeech2' or hp.model.lower() == 'lightspeech')
        dataset_train = datasets.TrainDatasets(hp.train_script, hp, alignment_pred=alignment_pred, pitch_pred=hp.pitch_pred, energy_pred=hp.energy_pred, accent_emb=hp.accent_emb)
        collate_fn = datasets.collate_fn
    if hp.batch_size is not None:
        sampler = datasets.NumBatchSampler(dataset_train, hp.batch_size)
    elif hp.max_seqlen is not None:
        sampler = datasets.LengthsBatchSampler(dataset_train, hp.max_seqlen, hp, hp.lengths_file,shuffle=True, shuffle_one_time=False)

    train_sampler = datasets.DistributedSamplerWrapper(sampler) if args.n_gpus > 1 else sampler
    dataloader = DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=8, collate_fn=collate_fn)

    for epoch in range(start_epoch, hp.max_epoch):
        start_time = time.time()
        step = train_loop(model, optimizer, step, epoch, args, hp, rank, dataloader)
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
        model = SQFastSpeech2(hp=hp, src_vocab=hp.vocab_size, trg_vocab=hp.mel_dim, d_model_encoder=hp.d_model_encoder, N_e=hp.n_layer_encoder,
                            n_head_encoder=hp.n_head_encoder, ff_conv_kernel_size_encoder=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.concat_after_encoder,
                            d_model_decoder=hp.d_model_decoder, N_d=hp.n_layer_decoder, n_head_decoder=hp.n_head_decoder,
                            ff_conv_kernel_size_decoder=hp.ff_conv_kernel_size_decoder, concat_after_decoder=hp.concat_after_decoder, dropout_variance_adaptor=hp.dropout_variance_adaptor,
                            reduction_rate=hp.reduction_rate, dropout=hp.dropout, dropout_postnet=0.5, 
                            n_bins=hp.nbins, f0_min=hp.f0_min, f0_max=hp.f0_max, energy_min=hp.energy_min, energy_max=hp.energy_max, pitch_pred=hp.pitch_pred, energy_pred=hp.energy_pred,
                            accent_emb=hp.accent_emb,
                            output_type=hp.output_type, num_group=hp.num_group, multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim, spk_emb_architecture=hp.spk_emb_architecture)
    
    model.apply(init_weight)
    model.train()

    if hp.is_multi_speaker and hp.pretrain_model is not None:
    #if hp.pretrain_model is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        loaded_dict = load_model(hp.pretrain_model, map_location=map_location)
        model.load_state_dict(loaded_dict, strict=False)
        print(f'{hp.pretrain_model} is loaded')
        del loaded_dict
        torch.cuda.empty_cache()
    
    max_lr = 1e-3
    if hp.optimizer.lower() == 'radam':
        import radam
        optimizer = radam.RAdam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)

    print(model)
    model = model.to(rank)
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
        #print('191-200')
        #loaded_dict = load_model("models.nwork1/checkpoints.FastSpeech2.sps_half.melgan_16kHz_spkid_wopretrain/network.average_epoch191-epoch200")

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
    train_epoch(model, optimizer, step, start_epoch, args, hp, rank)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', type=str, default='hparams.py')
    args = parser.parse_args()
    hp_file = args.hp_file

    hp.configure(hp_file)
    fill_variables(hp)
    log_config(hp)

    assert hp.architecture == 'text-mel', f'invalid architecture {hp.architecture}'
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
