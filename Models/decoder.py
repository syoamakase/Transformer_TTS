#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.layers import DecoderLayer
from Models.modules import PositionalEncoder
from Models.prenets import DecoderPreNet

class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args

def repeat(N, fn):
    """Repeat module N times.
    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn() for _ in range(N)])


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, ff_conv_kernel_size, concat_after_decoder, dropout,
                 dropout_prenet=0.5, multi_speaker=False, spk_emb_dim=None, output_type=None):
        super().__init__()
        self.N = N
        self.heads = heads
        self.output_type = output_type
        # self.embed = nn.Linear(vocab_size, d_model)
        # self.num_class = num_class
        self.decoder_prenet = DecoderPreNet(vocab_size, d_model, p=dropout_prenet, output_type=output_type)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = repeat(N, lambda: DecoderLayer(d_model, heads, ff_conv_kernel_size, dropout, concat_after_decoder, multi_speaker, spk_emb_dim))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask, spk_emb=None, attn_detach=True):
        x = self.decoder_prenet(trg)
        if self.output_type:
            x = x.sum(dim=2)
        x = self.pe(x)
        b, t1, _ = x.shape
        b, t2, _ = e_outputs.shape
        attns_1 = torch.zeros((b, self.N, self.heads, t1, t1), device=x.device)  # []
        attns_2 = torch.zeros((b, self.N, self.heads, t1, t2), device=x.device)  # []
        for i in range(self.N):
            x, attn_1, attn_2 = self.layers[i](x, e_outputs, src_mask, trg_mask, spk_emb)
            attns_1[:, i] = attn_1.detach() if attn_detach else attn_1
            attns_2[:, i] = attn_2.detach() if attn_detach else attn_2
        return self.norm(x), attns_1, attns_2

class Tacotron2Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_model_e, reduction_rate, conv_kernel_size=31,
                 dropout_prenet=0.5, multi_speaker=False, spk_emb_dim=None, zoneout_rate=0.1):
        super(Tacotron2Decoder, self).__init__()
        self.d_model = d_model
        self.zoneout_rate = zoneout_rate
        self.reduction_rate = reduction_rate
        self.multi_speaker = multi_speaker

        if multi_speaker:
            # xvector
            if spk_emb_dim == 512:
                self.speaker_embeddings = nn.Linear(spk_emb_dim, d_model)
            else:
                self.speaker_embeddings = nn.Embedding(spk_emb_dim, d_model)
        # multi decoder
        #self.speaker_L_input = nn.Linear(256, 512)

        d_model_2 = d_model * 2
        d_model_4 = d_model * 4
        self.L_spkr2s1 = nn.Linear(d_model, d_model_4)
        self.L_spkr2s2 = nn.Linear(d_model, d_model_4)
        self.speaker_L_l1_es = nn.Linear(d_model, d_model_4, bias=False)

        self.L_l1_ys = nn.Linear(d_model, d_model_4 * 4, bias=False)
        self.L_l1_ss = nn.Linear(d_model_4, d_model_4 * 4, bias=False)
        self.L_l1_gs = nn.Linear(d_model_2, d_model_4 * 4)
        self.L_l2_is = nn.Linear(d_model_4, d_model_4 * 4, bias=False)
        self.L_l2_ss = nn.Linear(d_model_4, d_model_4 * 4)

        #self.FrameProj = nn.Linear(1024 + 512, NUM_MELS * REDUCTION_RATE, bias=False)
        self.FrameProj = nn.Linear(d_model_4 + d_model_2, vocab_size * reduction_rate)
        #self.TokenProj = nn.Linear(1024 + 512, REDUCTION_RATE, bias=False)
        self.TokenProj = nn.Linear(d_model_4 + d_model_2, reduction_rate)
        # prenet
        self.Prenet1 = nn.Linear(vocab_size, d_model)
        self.Prenet2 = nn.Linear(d_model, d_model)
        # attention
        self.AttentionConv = nn.Conv1d(1, 32, conv_kernel_size, stride=1, padding=15, bias=False)
        self.AttentionConvProj = nn.Linear(32, 128, bias=False)
        self.AttentionEncoderProj = nn.Linear(d_model_e, 128)
        self.AttentionDecoderProj = nn.Linear(d_model_4, 128, bias=False)
        self.AttentionSelfProj = nn.Linear(128, 1, bias=False)

        self.dropout = nn.Dropout(dropout_prenet)

    def forward(self, meltarget, e_outputs, speaker_emb=None):
        target_length = meltarget.shape[1]
        input_length = e_outputs.shape[1]
        batch_size = e_outputs.shape[0]
        device = e_outputs.device

        decoder_steps = target_length // self.reduction_rate

        prev_prediction = torch.zeros((batch_size, self.vocab_size), device=device, requires_grad=False)
        alpha = torch.zeros((batch_size, input_length), device=device, requires_grad=False)
        cumulate_alpha = torch.zeros((batch_size, input_length), device=device, requires_grad=False)
        attention_weight = torch.zeros((batch_size, 500, input_length), device=device)

        if self.multi_speaker:
            spkr_embeds_dec = self.speaker_embeddings(speaker_emb)
            s1 = F.softsign(self.L_spkr2s1(spkr_embeds_dec))
            s2 = F.softsign(self.L_spkr2s2(spkr_embeds_dec))
        else:
            s1 = torch.zeros((batch_size, self.d_model*4), device=device, requires_grad=False)
            s2 = torch.zeros((batch_size, self.d_model*4), device=device, requires_grad=False)
        c1 = torch.zeros((batch_size, self.d_model*4), device=device, requires_grad=False)
        c2 = torch.zeros((batch_size, self.d_model*4), device=device, requires_grad=False)

        frame_prediction_sent = None
        token_prediction_sent = None

        for step in range(decoder_steps):
            attconv = self.AttentionConv(cumulate_alpha.unsqueeze(1))
            attconv = self.AttentionConvProj(attconv.transpose(1, 2)[:, :input_length, :])
            encproj = self.AttentionEncoderProj(e_outputs)
            decproj = self.AttentionDecoderProj(s2).unsqueeze(1)

            e = F.tanh(decproj + encproj + attconv)
            eproj = self.AttentionSelfProj(e).squeeze(2)
            eproj_nonlin = (eproj - eproj.max(1)[0].unsqueeze(1)).exp()
            alpha = eproj_nonlin / eproj_nonlin.sum(dim=1, keepdim=True)
            attention_weight[:, step] = alpha.detach()
            cumulate_alpha = cumulate_alpha + alpha
            g = (alpha.unsqueeze(2) * e_outputs).sum(dim=1)

            pre1 = F.relu(self.Prenet1(prev_prediction))
            pre1_drop = self.dropout(pre1)
            pre2 = F.relu(self.Prenet2(pre1_drop))
            pre2_drop = self.dropout(pre2)

            # My Zoneout LSTM Cell
            # layer 1
            if self.multi_speaker:
                rec_input = self.L_l1_ys(pre2_drop) + self.L_l1_ss(s1) + self.L_l1_gs(g) + F.softsign(self.speaker_L_l1_es(spkr_embeds_dec))
            else:
                rec_input = self.L_l1_ys(pre2_drop) + self.L_l1_ss(s1) + self.L_l1_gs(g)
            ingate, forgetgate, cellgate, outgate = rec_input.chunk(4, 1)
            half = 0.5
            ingate = F.tanh(ingate * half) * half + half
            forgetgate = F.tanh(forgetgate * half) * half + half
            cellgate = F.tanh(cellgate)
            outgate = F.tanh(outgate * half) * half + half
            # zoneout
            random_tensor = torch.rand((batch_size, self.d_model*4), device=device, requires_grad=False) + self.zoneout_rate
            mask_tensor = torch.floor(random_tensor)
            mask_tensor_complement = torch.ones((batch_size, self.d_model*4), device=device, requires_grad=False) - mask_tensor
            c_tmp = (forgetgate * c1) + (ingate * cellgate)
            c_next = mask_tensor * c1 + mask_tensor_complement * c_tmp
            h_tmp = outgate * F.tanh(c_next)
            h_next = mask_tensor * s1 + mask_tensor_complement * h_tmp
            s1 = h_next
            c1 = c_next

            # layer 2
            rec_input = self.L_l2_is(s1) + self.L_l2_ss(s2)
            ingate, forgetgate, cellgate, outgate = rec_input.chunk(4, 1)
            half = 0.5
            ingate = F.tanh(ingate * half) * half + half
            forgetgate = F.tanh(forgetgate * half) * half + half
            cellgate = F.tanh(cellgate)
            outgate = F.tanh(outgate * half) * half + half
            # zoneout
            random_tensor = torch.rand((batch_size, self.d_model*4), device=device, requires_grad=False) + self.zoneout_rate
            mask_tensor = torch.floor(random_tensor)
            mask_tensor_complement = torch.ones((batch_size, self.d_model*4), device=device, requires_grad=False) - mask_tensor
            c_tmp = (forgetgate * c2) + (ingate * cellgate)
            c_next = mask_tensor * c2 + mask_tensor_complement * c_tmp
            h_tmp = outgate * F.tanh(c_next)
            h_next = mask_tensor * s2 + mask_tensor_complement * h_tmp
            s2 = h_next
            c2 = c_next

            proj_input = torch.cat((s2, g), dim=1)
            frame_prediction = self.FrameProj(proj_input)
            token_prediction = self.TokenProj(proj_input)

            prev_prediction = meltarget[:, step * self.reduction_rate + self.reduction_rate - 1, :]

            if frame_prediction_sent is None:
                frame_prediction_sent = frame_prediction.view(batch_size, self.reduction_rate, self.vocab_size)
            else:
                frame_prediction_sent = torch.cat((frame_prediction_sent, frame_prediction.view(batch_size, self.reduction_rate, self.vocab_size)), dim=1)

            if token_prediction_sent is None:
                token_prediction_sent = token_prediction
            else:
                token_prediction_sent = torch.cat((token_prediction_sent, token_prediction), dim=1)

        return frame_prediction_sent, token_prediction_sent, attention_weight

    def synthesize(self, e_outputs, text_lengths=None, speaker_emb=None):
        device = e_outputs.device
        input_length = e_outputs.shape[1]
        batch_size = e_outputs.shape[0]
        zoneout_rate = 0

        e_mask = torch.ones((batch_size, input_length), device=device, requires_grad=False)
        if text_lengths is not None:
            for i, tmp in enumerate(text_lengths):
                if tmp < input_length:
                    e_mask.data[i, tmp:] = 0.0

        prev_prediction = torch.zeros((batch_size, self.vocab_size), device=device, requires_grad=False)
        alpha = torch.zeros((batch_size, input_length), device=device, requires_grad=False)
        cumulate_alpha = torch.zeros((batch_size, input_length), device=device, requires_grad=False)
        attention_weight = torch.zeros((batch_size, 500, input_length), device=device)

        if self.imulti_speaker:
            spkr_embeds_dec = self.speaker_embeddings(speaker_emb)
            s1 = F.softsign(self.L_spkr2s1(spkr_embeds_dec))
            s2 = F.softsign(self.L_spkr2s2(spkr_embeds_dec))
        else:
            s1 = torch.zeros((batch_size, self.d_model*4), device=device, requires_grad=False)
            s2 = torch.zeros((batch_size, self.d_model*4), device=device, requires_grad=False)

        c1 = torch.zeros((batch_size, self.d_model*4), device=device, requires_grad=False)
        c2 = torch.zeros((batch_size, self.d_model*4), device=device, requires_grad=False)

        frame_prediction_sent = None
        token_prediction_sent = None

        # end_frames = torch.zeros((batch_size), device=DEVICE)
        end_tail = 4
        end_detected = False

        #for step in range(decoder_steps):
        for step in range(500):
            attconv = self.AttentionConv(cumulate_alpha.unsqueeze(1))
            attconv = self.AttentionConvProj(attconv.transpose(1, 2)[:, :input_length, :])
            encproj = self.AttentionEncoderProj(e_outputs)
            decproj = self.AttentionDecoderProj(s2).unsqueeze(1)
            e = F.tanh(decproj + encproj + attconv)
            eproj = self.AttentionSelfProj(e).squeeze(2)
            eproj_nonlin = eproj.exp() # * e_mask
            alpha = eproj_nonlin / eproj_nonlin.sum(dim=1, keepdim=True)
            attention_weight[:, step] = alpha.detach()
            cumulate_alpha = cumulate_alpha + alpha
            g = (alpha.unsqueeze(2) * e_outputs).sum(dim=1)

            #prenet_output = F.dropout(self.Prenet2(F.dropout(self.Prenet1(prev_prediction), p = DROPOUT_RATE, training = True)), p = DROPOUT_RATE, training = True)
            pre1 = F.relu(self.Prenet1(prev_prediction))
            # pre1_drop = F.dropout(pre1, p=0.0, training=self.training)
            pre2 = F.relu(self.Prenet2(pre1))
            # pre2_drop = F.dropout(pre2, p=0.0, training=self.training)

            # My Zoneout LSTM Cell
            # layer 1
            if self.multi_speaker:
                rec_input = self.L_l1_ys(pre2) + self.L_l1_ss(s1) + self.L_l1_gs(g) + F.softsign(self.speaker_L_l1_es(spkr_embeds_dec))
            else:
                rec_input = self.L_l1_ys(pre2) + self.L_l1_ss(s1) + self.L_l1_gs(g)
            ingate, forgetgate, cellgate, outgate = rec_input.chunk(4, 1)
            half = 0.5
            ingate = F.tanh(ingate * half) * half + half
            forgetgate = F.tanh(forgetgate * half) * half + half
            cellgate = F.tanh(cellgate)
            outgate = F.tanh(outgate * half) * half + half
            # zoneout
            random_tensor = torch.rand((batch_size, 1024), device=device, requires_grad=False) + zoneout_rate
            mask_tensor = torch.floor(random_tensor)
            mask_tensor_complement = torch.ones((batch_size, 1024), device=device, requires_grad=False) - mask_tensor
            c_tmp = (forgetgate * c1) + (ingate * cellgate)
            c_next = mask_tensor * c1 + mask_tensor_complement * c_tmp
            h_tmp = outgate * F.tanh(c_next)
            h_next = mask_tensor * s1 + mask_tensor_complement * h_tmp
            s1 = h_next
            c1 = c_next

            # layer 2
            rec_input = self.L_l2_is(s1) + self.L_l2_ss(s2)
            ingate, forgetgate, cellgate, outgate = rec_input.chunk(4, 1)
            half = 0.5
            ingate = F.tanh(ingate * half) * half + half
            forgetgate = F.tanh(forgetgate * half) * half + half
            cellgate = F.tanh(cellgate)
            outgate = F.tanh(outgate * half) * half + half
            # zoneout
            random_tensor = torch.rand((batch_size, self.d_model*4), device=device, requires_grad=False) + 0.0
            mask_tensor = torch.floor(random_tensor)
            mask_tensor_complement = torch.ones((batch_size, self.d_model*4), device=device, requires_grad=False) - mask_tensor
            c_tmp = (forgetgate * c2) + (ingate * cellgate)
            c_next = mask_tensor * c2 + mask_tensor_complement * c_tmp
            h_tmp = outgate * F.tanh(c_next)
            h_next = mask_tensor * s2 + mask_tensor_complement * h_tmp
            s2 = h_next
            c2 = c_next

            proj_input = torch.cat((s2, g), dim=1)
            frame_prediction = self.FrameProj(proj_input)
            token_prediction = F.sigmoid(self.TokenProj(proj_input))

            prev_prediction = frame_prediction.view(batch_size, self.reduction_rate, self.vocab_size)[:, self.reduction_rate - 1, :]

            if frame_prediction_sent is None:
                frame_prediction_sent = frame_prediction.view(batch_size, self.reduction_rate, self.vocab_size)
            else:
                frame_prediction_sent = torch.cat((frame_prediction_sent, frame_prediction.view(batch_size, self.reduction_rate, self.vocab_size)), dim=1)

            if token_prediction_sent is None:
                token_prediction_sent = token_prediction
            else:
                token_prediction_sent = torch.cat((token_prediction_sent, token_prediction), dim=1)

            # sentence end detected
            # for idx in range(batch_size):
            #     # if torch.mean(token_prediction[idx, :]) > 0.3 and end_frames[idx] == 0:
                    # end_frames[idx] = step * hp.reduction_rate
            if ((torch.mean(token_prediction[0, :]) > 0.5 or alpha[0, -1] > 0.85) and step > 10) or end_detected:
                end_detected = True
                end_tail -= 1
                if end_tail < 1:
                    break 

        return frame_prediction_sent, token_prediction_sent, attention_weight

    def decode_one_step(self):
        pass
