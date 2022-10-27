# -*- coding: utf-8 -*-
import argparse
import copy
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import hparams as hp
from utils.utils import log_config, fill_variables, init_weight, get_learning_rate, load_model
from Models.transformer import Transformer
from datasets import datasets_transformer

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
    """Create masks for transformer

    Args:
        src_pos (LongTensor) [B x T]: The positions of input of encoder
        trg_pos (LongTensor) [B x T]: The positions of input/output of decoder
        src_pad (int): The padding value of `src_pos` (default=0)
        trg_pad (int): The padding value of `trg_pos` (default=0)
    Returns:
        (Tensor, Tensor). masks for encoder and decoder

    """
    src_mask = (src_pos != src_pad).unsqueeze(-2)
    if trg_pos is not None:
        trg_mask = (trg_pos != trg_pad).unsqueeze(-2)
        size = trg_pos.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size).to(trg_pos.device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

def align_loss_attn(attn, src_len, trg_len, alignment=None, diagonal=False):
    # TODO: alignment mode
    assert (diagonal is True and alignment is None) or (diagonal is False and alignment is not None)

    B = len(attn)
    import pdb; pdb.set_trace()
    loss_attn = 0
    for i in range(B):
        if diagonal:
            assert src_len[i] == trg_len[i]
            results_attns = torch.diag(torch.ones(src_len[i])).to(attn[i])
            loss_attn += F.l1_loss(attn[i], results_attns)
    return loss_attn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', type=str, default='hparams.py')
    args = parser.parse_args()
    hp_file = args.hp_file

    hp.configure(hp_file)
    fill_variables(hp)
    log_config(hp)

    model = Transformer(hp=hp, src_vocab=hp.vocab_size, trg_vocab=hp.mel_dim,
                        d_model_encoder=hp.d_model_encoder, N_e=hp.n_layer_encoder, n_head_encoder=hp.n_head_encoder,
                        ff_conv_kernel_size_encoder=hp.ff_conv_kernel_size_encoder, concat_after_encoder=hp.concat_after_encoder,
                        d_model_decoder=hp.d_model_decoder, N_d=hp.n_layer_decoder, n_head_decoder=hp.n_head_decoder,
                        ff_conv_kernel_size_decoder=hp.ff_conv_kernel_size_decoder, concat_after_decoder=hp.concat_after_decoder,
                        reduction_rate=hp.reduction_rate, dropout=hp.dropout, dropout_prenet=hp.dropout_prenet, dropout_postnet=hp.dropout_postnet,
                        multi_speaker=hp.is_multi_speaker, spk_emb_dim=hp.spk_emb_dim, spk_emb_architecture=hp.spk_emb_architecture)

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
    scaler = torch.cuda.amp.GradScaler()

    # if hp.is_multi_speaker:
    #     model.load_state_dict(torch.load(hp.pretrain_model), strict=False)

    max_lr = 1e-3
    warmup_step = hp.warmup_step  # 4000
    warmup_factor = hp.warmup_factor  # 1
    if hp.optimizer.lower() == 'radam':
        import radam
        optimizer = radam.RAdam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    elif hp.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)

    save_dir = hp.save_dir  # save dir name
    os.makedirs(save_dir, exist_ok=True)
    print(model)
    print(optimizer)
    # if hp_file != f'{save_dir}/hparams.py':
    #     shutil.copyfile(hp_file, f'{save_dir}/hparams.py')
    # writer = SummaryWriter(f'{hp.log_dir}/logs/{hp.comment}')

    dataset_train = datasets_transformer.TrainDatasets(hp.train_script, hp)
    collate_fn_transformer = datasets_transformer.collate_fn

    if hp.batch_size is not None:
        sampler = datasets_transformer.NumBatchSampler(dataset_train, hp.batch_size)
    elif hp.max_seqlen is not None:
        sampler = datasets_transformer.LengthsBatchSampler(dataset_train, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False)

    assert (hp.batch_size is None) != (hp.max_seqlen is None)

    if hp.loaded_epoch is not None:
        start_epoch = hp.loaded_epoch
        load_dir = hp.loaded_dir
        print('epoch {} loaded'.format(hp.loaded_epoch))
        model.load_state_dict(load_model("{}".format(os.path.join(load_dir, 'network.epoch{}'.format(hp.loaded_epoch)))))
        
        loaded_dict = torch.load("{}".format(os.path.join(load_dir, 'network.optimizer.epoch{}'.format(hp.loaded_epoch))))
        optimizer.load_state_dict(loaded_dict)
        #optimizer.load_state_dict(torch.load("{}".format(os.path.join(load_dir, 'network.optimizer.epoch{}'.format(hp.loaded_epoch)))))

        dataloader = DataLoader(dataset_train, batch_sampler=sampler, num_workers=1, collate_fn=collate_fn_transformer)
        #step = hp.loaded_epoch * len(dataloader)
        step = loaded_dict['state'][0]['step'] * hp.accum_grad
    else:
        start_epoch = 0
        step = 1

    for epoch in range(start_epoch, hp.max_epoch):
        dataloader = DataLoader(dataset_train, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn_transformer)
        start_time = time.time()

        for d in dataloader:
            if hp.optimizer.lower() != 'radam':
                # lr = get_learning_rate(step, hp.d_model_decoder, hp.warmup_factor, hp.warmup_step)
                #lr = get_learning_rate(step//hp.accum_grad+1, d_model=hp.d_model_decoder, warmup_factor=hp.warmup_factor, warmup_step=hp.warmup_step)
                lr = get_learning_rate(step, d_model=hp.d_model_decoder, warmup_factor=hp.warmup_factor, warmup_step=hp.warmup_step)
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
            if spk_emb is not None:
                spk_emb = spk_emb.to(DEVICE)
                ############################################### 10/4
                spk_emb = F.normalize(spk_emb)
                #################################

            batch_size = mel.shape[0]
            if hp.decoder_type.lower() == 'transformer':
                if hp.reduction_rate > 1:
                    mel_input = mel[:,:-hp.reduction_rate:hp.reduction_rate, :]
                    pos_mel = pos_mel[:,:-hp.reduction_rate:hp.reduction_rate]
                    mel_lengths = (mel_lengths - hp.reduction_rate) / hp.reduction_rate
                else:
                    mel_input = mel[:,:-1,:]
                    pos_mel = pos_mel[:,:-1]
            else:
                mel_input = mel

            src_mask, trg_mask = create_masks(pos_text, pos_mel)

            # if hp.CTC_training:
            #     outputs_prenet, outputs_postnet, outputs_stop_token, attn_enc, attn_dec_dec, attn_dec_enc, ctc_outputs = model(text, mel_input, src_mask, trg_mask, spk_emb)
            # else:
            with torch.cuda.amp.autocast(hp.amp):
                outputs_prenet, outputs_postnet, outputs_stop_token, attn_enc, attn_dec_dec, attn_dec_enc = model(text, mel_input, src_mask, trg_mask, spk_emb)
                attn_dec_dec = attn_dec_dec.cpu()#.numpy()
                attn_dec_enc = attn_dec_enc.cpu()#.numpy()
                attn_enc = attn_enc.cpu()#.numpy()
                optimizer.zero_grad()
                # import pdb; pdb.set_trace()
                if hp.reduction_rate > 1:
                    b,t,c = outputs_prenet.shape
                    outputs_prenet = outputs_prenet.reshape(b, t*hp.reduction_rate, int(c//hp.reduction_rate))
                    outputs_postnet = outputs_postnet.reshape(b, t*hp.reduction_rate, int(c//hp.reduction_rate))
                    outputs_stop_token = outputs_stop_token.reshape(b, t*hp.reduction_rate)

                loss = 0.0
                mel_loss = nn.L1Loss()(outputs_prenet, mel[:,hp.reduction_rate:,:])
                post_mel_loss = nn.L1Loss()(outputs_postnet, mel[:,hp.reduction_rate:,:])
                # loss_token = F.binary_cross_entropy_with_logits(outputs_stop_token, stop_token[:,hp.reduction_rate:], size_average=True, pos_weight=torch.tensor(hp.positive_weight))
                loss_token = F.binary_cross_entropy_with_logits(outputs_stop_token, stop_token[:,hp.reduction_rate:], reduction='mean', pos_weight=torch.tensor(hp.positive_weight))
                loss = mel_loss + post_mel_loss
                loss += loss_token

                # if hp.CTC_training:
                #     ctc_outputs = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                #     loss_ctc = F.ctc_loss(ctc_outputs, text, mel_lengths, text_lengths, blank=0)
                #     loss += 0.2 * loss_ctc
                #     writer.add_scalar("Loss/train_ctc_loss", loss_ctc, step)

                # if step % hp.save_attention_per_step == 0:
                #     for n in range(hp.n_layer_encoder):
                #         for head in range(hp.n_head_encoder):
                #             writer.add_image(f'attn_enc_{n}_{head}', attn_enc[0][n][head], step, dataformats='HW')
                #     for n in range(hp.n_layer_decoder):
                #         for head in range(hp.n_head_encoder):
                #             writer.add_image(f'attn_dec_dec_{n}_{head}', attn_dec_dec[0][n][head], step, dataformats='HW')
                #             writer.add_image(f'attn_dec_enc_{n}_{head}', attn_dec_enc[0][n][head], step, dataformats='HW')
                print(f'step {step}')
                print(f'loss_token = {loss_token.item()}')
                print(f'loss_frame_before = {mel_loss.item()}')
                print(f'loss_frame_after = {post_mel_loss.item()}')
                print(f'loss_token = {loss_token.item()}')
                print(f'loss_total = {loss.item()}')
                print(f'batch size = {batch_size}')
                print(f'loss = {loss.item()}')
                print(f'lr = {lr}')

                step += 1

                if hp.amp:
                    loss /= hp.accum_grad
                    # import pdb; pdb.set_trace()
                    scaler.scale(loss).backward()
                    if step % hp.accum_grad == 0:
                        if hp.clip is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss /= hp.accum_grad
                    loss.backward()
                    # backward
                    if step % hp.accum_grad == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                        optimizer.step()
                sys.stdout.flush()
                if torch.isnan(loss):
                    import pdb;pdb.set_trace()
                assert not torch.isnan(loss), f'loss is nan'
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
        print(f'elapsed time {time.time() - start_time}')
