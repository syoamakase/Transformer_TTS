# -*- coding: utf-8 -*-
import argparse
import copy
import os
import numpy as np
import random
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm

from utils import hparams as hp
from utils.utils import log_config, fill_variables, init_weight, get_learning_rate, load_model
from Models.transformer import Transformer
import datasets

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
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0).to(DEVICE)
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

def synthesize(test_script, model):
    model.eval()

    dataset_train = datasets.get_dataset(hp.test_script)
    collate_fn_transformer = datasets.collate_fn
    sampler = datasets.NumBatchSampler(dataset_train, 1)

    for d in dataloader: 
        text, mel, pos_text, pos_mel, text_lengths, mel_lengths, stop_token, spk_emb = d

        text = text.to(DEVICE)
        mel = mel.to(DEVICE)
        pos_text = pos_text.to(DEVICE)
        pos_mel = pos_mel.to(DEVICE)
        text_lengths = text_lengths.to(DEVICE)
        stop_token = stop_token.to(DEVICE)

        batch_size = mel.shape[0]
        dummy_input = mel[:,0,:].unsqueeze(1) #torch.zeros((1, 1, 80), device=DEVICE, dtype=torch.float)

        pos_mel = torch.arange(start=1,end=2,dtype=torch.long, device=DEVICE).unsqueeze(0)
        src_mask, trg_mask = create_masks(pos_text, pos_mel)
        mel_input = dummy_input
        outputs_prenet, outputs_postnet, outputs_stop_token = model(text, mel_input, src_mask, trg_mask)
        print(outputs_stop_token[0, -1].item())
        mel_input = outputs_postnet
        with torch.no_grad():
            for i in range(2,300):
                mel_input = torch.cat((dummy_input, mel_input), dim=1)
                pos_mel = torch.arange(start=1,end=i+1,dtype=torch.long, device=DEVICE).unsqueeze(0)
                src_mask, trg_mask = create_masks(pos_text, pos_mel)
                outputs_prenet, outputs_postnet, outputs_stop_token = model(text, mel_input, src_mask, trg_mask)
                print(outputs_stop_token[0, -1].item())
                mel_input = outputs_postnet
            sys.stdout.flush()

    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', type=str, default='hparams.py')
    args = parser.parse_args()
    hp_file = args.hp_file

    hp.configure(hp_file)
    fill_variables()
    log_config()

    save_dir = hp.save_dir # save dir name
    model = Transformer(src_vocab=hp.vocab_size, trg_vocab=hp.mel_dim, d_model_encoder=hp.d_model_encoder, N_e=hp.n_layer_encoder,
                        n_head_encoder=hp.n_head_encoder, ff_conv_kernel_size_encoder=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.ff_conv_kernel_size_encoder,
                        d_model_decoder=hp.d_model_decoder, N_d=hp.n_layer_decoder, n_head_decoder=hp.n_head_decoder,
                        ff_conv_kernel_size_decoder=hp.ff_conv_kernel_size_decoder, concat_after_decoder=hp.concat_after_decoder,
                        reduction_rate=hp.reduction_rate, dropout=hp.dropout, CTC_training=hp.CTC_training)

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
    warmup_step = hp.warmup_step # 4000
    warmup_factor = hp.warmup_factor #1.0
    if hp.optimizer.lower() == 'radam':
        import radam
        optimizer = radam.RAdam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)

    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(f'{hp.log_dir}/logs/{hp.comment}')

    dataset_train = datasets.get_dataset(hp.train_script)
    collate_fn_transformer = datasets.collate_fn
    if hp.batch_size is not None:
        sampler = datasets.NumBatchSampler(dataset_train, hp.batch_size)
    elif hp.max_seqlen is not None:
        #sampler = dataset.LengthsBatchSampler(datasets, hp.max_seqlen, hp.lengths_file, shuffle=False, shuffle_one_time=True)
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

        #scheduler.step(epoch)
        #pbar = tqdm(dataloader)
        #for d in pbar:
        for d in dataloader: 
            if hp.optimizer.lower() != 'radam':
                lr = get_learning_rate(step, hp.d_model_decoder, hp.warmup_factor, hp.warmup_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            text, mel, pos_text, pos_mel, text_lengths, mel_lengths, stop_token, spk_emb = d

            text = text.to(DEVICE)
            mel = mel.to(DEVICE)
            pos_text = pos_text.to(DEVICE)
            pos_mel = pos_mel.to(DEVICE)
            mel_lengths = mel_lengths.to(DEVICE)
            text_lengths = text_lengths.to(DEVICE)
            stop_token = stop_token.to(DEVICE)

            batch_size = mel.shape[0]
            if hp.reduction_rate > 1:
                mel_input = mel[:,:-hp.reduction_rate:hp.reduction_rate,:]
                pos_mel = pos_mel[:,:-hp.reduction_rate:hp.reduction_rate]
                mel_lengths = (mel_lengths - hp.reduction_rate) / hp.reduction_rate
            else:
                mel_input = mel[:,:-1,:]
                pos_mel = pos_mel[:,:-1]

            src_mask, trg_mask = create_masks(pos_text, pos_mel)

            if hp.CTC_training:
                outputs_prenet, outputs_postnet, outputs_stop_token, attn_enc, attn_dec_dec, attn_dec_enc, ctc_outputs = model(text, mel_input, src_mask, trg_mask)
            else:
                outputs_prenet, outputs_postnet, outputs_stop_token, attn_enc, attn_dec_dec, attn_dec_enc = model(text, mel_input, src_mask, trg_mask)
            attn_dec_dec = attn_dec_dec.cpu().numpy()
            attn_dec_enc = attn_dec_enc.cpu().numpy()
            attn_enc = attn_enc.cpu().numpy()
            optimizer.zero_grad()
            if hp.reduction_rate > 1:
                b,t,c = outputs_prenet.shape
                outputs_prenet = outputs_prenet.reshape(b, t*hp.reduction_rate, int(c//hp.reduction_rate))
                outputs_postnet = outputs_postnet.reshape(b, t*hp.reduction_rate, int(c//hp.reduction_rate))
                outputs_stop_token = outputs_stop_token.reshape(b, t*hp.reduction_rate)

            loss = 0.0
            mel_loss = nn.L1Loss()(outputs_prenet, mel[:,hp.reduction_rate:,:])
            post_mel_loss = nn.L1Loss()(outputs_postnet, mel[:,hp.reduction_rate:,:])
            loss_token = F.binary_cross_entropy_with_logits(outputs_stop_token, stop_token[:,hp.reduction_rate:], size_average=True, pos_weight=torch.tensor(5.0))
            loss = mel_loss + post_mel_loss
            loss += loss_token

            if hp.CTC_training:
                # print(ctc_outputs.shape)
                # print(text.shape)
                # print(mel_lengths.shape)
                # print(mel_lengths)
                # print(text_lengths.shape)
                # print(text_lengths)
                ctc_outputs = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                loss_ctc = F.ctc_loss(ctc_outputs, text, mel_lengths, text_lengths, blank=0)
                loss += 0.2 * loss_ctc
                writer.add_scalar("Loss/train_ctc_loss", loss_ctc, step)

            writer.add_scalar("Loss/train_post_mel", post_mel_loss, step)
            writer.add_scalar("Loss/train_pre_mel", mel_loss, step)
            writer.add_scalar("Loss/train_token", loss_token, step)
            writer.add_scalar("Loss/train_all_loss", loss, step)
            if step % hp.save_attention_per_step == 0:
                for n in range(hp.n_layer_encoder):
                    for head in range(hp.n_head_encoder):
                        writer.add_image(f'attn_enc_{n}_{head}', attn_enc[0][n][head], step, dataformats='HW')
                for n in range(hp.n_layer_decoder):
                    for head in range(hp.n_head_encoder):
                        writer.add_image(f'attn_dec_dec_{n}_{head}', attn_dec_dec[0][n][head], step, dataformats='HW')
                        writer.add_image(f'attn_dec_enc_{n}_{head}', attn_dec_enc[0][n][head], step, dataformats='HW')
            #print('lr = {}'.format(lr))
            print(f'loss_token = {loss_token.item()}')
            print(f'loss_frame_before = {mel_loss.item()}')
            print(f'loss_frame_after = {post_mel_loss.item()}')
            print(f'loss_token = {loss_token.item()}')
            print(f'loss_total = {loss.item()}')
            print(f'batch size = {batch_size}')
            print(f'step {step}')
            step += 1

            print(f'loss = {loss.item()}')
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
            
        print('loss =', loss.item())
        print("EPOCH {} end".format(epoch+1))
