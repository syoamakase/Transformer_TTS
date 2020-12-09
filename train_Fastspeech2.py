# -*- coding: utf-8 -*-
import argparse
import copy
import filecmp
import os
import numpy as np
import random
import sys
import shutil
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm

from utils import hparams as hp
from utils.utils import log_config, fill_variables, init_weight, get_learning_rate, load_model
# from Models.transformer import Transformer
from Models.fastspeech2 import FastSpeech2
import datasets.datasets_fastspeech2 as datasets

random.seed(77)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nopeak_mask(size):
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

def create_masks(src_pos, trg_pos, src_pad=0, trg_pad=0):
    src_mask = (src_pos != src_pad).unsqueeze(-2)
    trg_mask = (trg_pos != trg_pad).unsqueeze(-2)
    # For general transformer
    #if trg_pos is not None:
    #    trg_mask = (trg_pos != trg_pad).unsqueeze(-2)
    #    size = trg_pos.size(1) # get seq_len for matrix
    #    np_mask = nopeak_mask(size)
    #    if trg_pos.is_cuda:
    #        np_mask.cuda()
    #    trg_mask = trg_mask & np_mask
    #else:
    #    trg_mask = None
    return src_mask, trg_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', type=str, default='hparams.py')
    args = parser.parse_args()
    hp_file = args.hp_file

    hp.configure(hp_file)
    fill_variables()
    log_config()

    model = FastSpeech2(src_vocab=hp.vocab_size, trg_vocab=hp.mel_dim, d_model_encoder=hp.d_model_encoder, N_e=hp.n_layer_encoder,
                        n_head_encoder=hp.n_head_encoder, ff_conv_kernel_size_encoder=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.ff_conv_kernel_size_encoder,
                        d_model_decoder=hp.d_model_decoder, N_d=hp.n_layer_decoder, n_head_decoder=hp.n_head_decoder,
                        ff_conv_kernel_size_decoder=hp.ff_conv_kernel_size_decoder, concat_after_decoder=hp.concat_after_decoder,
                        reduction_rate=hp.reduction_rate, dropout=hp.dropout, CTC_training=hp.CTC_training,
                        n_bins=hp.nbins, f0_min=hp.f0_min, f0_max=hp.f0_max, energy_min=hp.energy_min, energy_max=hp.energy_max, pitch_pred=hp.pitch_pred, energy_pred=hp.energy_pred,
                        output_type=hp.output_type, num_group=hp.num_group, multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.num_speaker)

    if hp.is_multi_speaker and hp.pretrain_model is not None:
        model.load_state_dict(torch.load(hp.pretrain_model), strict=False)

    # multi-gpu setup
    if torch.cuda.device_count() > 1:
        # multi-gpu configuration
        ngpu = torch.cuda.device_count()
        device_ids = list(range(ngpu))
        model = torch.nn.DataParallel(model, device_ids)
        model.cuda()
    else:
        model.to(DEVICE)

    model.apply(init_weight)
    model.train()

    max_lr = 1e-3
    warmup_step = hp.warmup_step
    warmup_factor = hp.warmup_factor
    if hp.optimizer.lower() == 'radam':
        import radam
        optimizer = radam.RAdam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)

    save_dir = hp.save_dir # save dir name
    os.makedirs(save_dir, exist_ok=True)
    if hp_file != f'{save_dir}/hparams.py' and not filecmp.cmp(hp_file, f'{save_dir}/hparams.py'):
        shutil.copyfile(hp_file, f'{save_dir}/hparams.py')
    writer = SummaryWriter(f'{hp.log_dir}/logs/{hp.comment}')

    if hp.output_type == 'softmax':
        dataset_train = datasets.VQWav2vecTrainDatasets(hp.train_script)
        collate_fn_transformer = datasets.collate_fn_vqwav2vec
    else:
        dataset_train = datasets.get_dataset(hp.train_script)
        collate_fn_transformer = datasets.collate_fn
    if hp.batch_size is not None:
        sampler = datasets.NumBatchSampler(dataset_train, hp.batch_size)
    elif hp.max_seqlen is not None:
        sampler = datasets.LengthsBatchSampler(dataset_train, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False)

    assert (hp.batch_size is None) != (hp.max_seqlen is None)

    if hp.loaded_epoch is not None:
        start_epoch = hp.loaded_epoch
        load_dir = hp.loaded_dir
        print('epoch {} loaded'.format(hp.loaded_epoch))
        model.load_state_dict(load_model("{}".format(os.path.join(load_dir, 'network.epoch{}'.format(hp.loaded_epoch)))))
        optimizer.load_state_dict(torch.load("{}".format(os.path.join(load_dir, 'network.optimizer.epoch{}'.format(hp.loaded_epoch)))))
    
        dataloader = DataLoader(dataset_train, batch_sampler=sampler, num_workers=1, collate_fn=collate_fn_transformer)
        step = hp.loaded_epoch * len(dataloader)
    else:
        start_epoch = 0
        step = 1

    for epoch in range(start_epoch, hp.max_epoch):
        dataloader = DataLoader(dataset_train, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn_transformer)

        #pbar = tqdm(dataloader)
        #for d in pbar:
        for d in dataloader: 
            if hp.optimizer.lower() != 'radam':
                lr = get_learning_rate(step, hp.d_model_decoder, hp.warmup_factor, hp.warmup_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                text, mel, pos_text, pos_mel, text_lengths, mel_lengths, stop_token, spk_emb, f0, energy, alignment = d

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
                alignment = alignment.to(DEVICE, non_blocking=True)

            batch_size = mel.shape[0]
            #if hp.reduction_rate > 1:
            #    mel_input = mel[:,:-hp.reduction_rate:hp.reduction_rate,:]
            #    pos_mel = pos_mel[:,:-hp.reduction_rate:hp.reduction_rate]
            #    mel_lengths = (mel_lengths - hp.reduction_rate) / hp.reduction_rate
            #else:
            #    mel_input = mel[:,:-1,:]
            #    pos_mel = pos_mel[:,:-1]

            src_mask, trg_mask = create_masks(pos_text, pos_mel)

            if hp.CTC_training:
                outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction = model(text, src_mask)
            else:
                # self.variance_adaptor(e_outputs, d_target, p_target, e_target, max_length)
                # x, src_mask, mel_mask=None, duration_target=None, pitch_target=None, energy_target=None, max_len=None):
                outputs_prenet, outputs_postnet, log_d_prediction, p_prediction, e_prediction, attn_enc, attn_dec = model(text, src_mask, trg_mask, alignment, f0, energy, spkr_emb=spk_emb)
            
            attn_dec_dec = attn_dec.cpu().numpy()
            # attn_dec_enc = attn_dec_enc.cpu().numpy()
            attn_enc = attn_enc.cpu().numpy()
            optimizer.zero_grad()
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
                writer.add_scalar("Acc/accuracy_1", acc1, step)
                writer.add_scalar("Acc/accuracy_2", acc2, step)
            else:
                mel_loss = nn.L1Loss()(outputs_prenet, mel)
                post_mel_loss = nn.L1Loss()(outputs_postnet, mel)
            #loss_token = F.binary_cross_entropy_with_logits(outputs_stop_token, stop_token[:,hp.reduction_rate:], size_average=True, pos_weight=torch.tensor(hp.positive_weight))
            
            duration_loss = nn.L1Loss()(log_d_prediction, torch.log(alignment.float() + 1)) # 1 = logoffset
            loss = mel_loss + post_mel_loss
            if hp.pitch_pred:
                f0_loss = nn.L1Loss()(p_prediction, f0)
                loss += f0_loss
                print(f'loss_f0 = {f0_loss.item()}')
                writer.add_scalar("Loss/train_f0", f0_loss, step)
            if hp.energy_pred:
                energy_loss = nn.L1Loss()(e_prediction, energy)
                loss += energy_loss
                print(f'loss_energy = {energy_loss.item()}')
                writer.add_scalar("Loss/train_energy", energy_loss, step)
            
            loss += duration_loss  #+ f0_loss + energy_loss

            if hp.CTC_training:
                ctc_outputs = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                loss_ctc = F.ctc_loss(ctc_outputs, text, mel_lengths, text_lengths, blank=0)
                loss += 0.2 * loss_ctc
                writer.add_scalar("Loss/train_ctc_loss", loss_ctc, step)

            writer.add_scalar("Loss/train_post_mel", post_mel_loss, step)
            writer.add_scalar("Loss/train_pre_mel", mel_loss, step)
            writer.add_scalar("Loss/train_duration", duration_loss, step)
            writer.add_scalar("Loss/train_all_loss", loss, step)
            if step % hp.save_attention_per_step == 0:
                for n in range(hp.n_layer_encoder):
                    for head in range(hp.n_head_encoder):
                        writer.add_image(f'attn_enc_{n}_{head}', attn_enc[0][n][head], step, dataformats='HW')
                for n in range(hp.n_layer_decoder):
                    for head in range(hp.n_head_encoder):
                        writer.add_image(f'attn_dec_{n}_{head}', attn_dec[0][n][head], step, dataformats='HW')
                        #writer.add_image(f'attn_dec_enc_{n}_{head}', attn_dec_enc[0][n][head], step, dataformats='HW')
            #print('lr = {}'.format(lr))
            print(f'loss_frame_before = {mel_loss.item()}')
            print(f'loss_frame_after = {post_mel_loss.item()}')
            print(f'loss_duration = {duration_loss.item()}')
            
            print(f'loss_total = {loss.item()}')
            print(f'batch size = {batch_size}')
            print(f'step {step}')
            step += 1

            loss.backward()
            # backward
            sys.stdout.flush()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch+1) >= (hp.max_epoch-10):
            torch.save(model.state_dict(), save_dir+"/network.epoch{}".format(epoch+1))
            #torch.save(optimizer.state_dict(), save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        elif (epoch+1) % hp.save_per_epoch >= (hp.save_per_epoch-10) or ((epoch+1) % hp.save_per_epoch == 0):
            torch.save(model.state_dict(), save_dir+"/network.epoch{}".format(epoch+1))
            # torch.save(optimizer.state_dict(), save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        
        if (epoch+1) % hp.save_per_epoch == 0:
            torch.save(optimizer.state_dict(), save_dir+"/network.optimizer.epoch{}".format(epoch+1))
            
        print("EPOCH {} end".format(epoch+1))
