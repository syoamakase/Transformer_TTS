import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from scipy.io.wavfile import read

from utils.stft import TacotronSTFT
from utils import hparams as hp

def main(hp, args):
    stft = TacotronSTFT(filter_length=hp.filter_length,
                        hop_length=hp.hop_length,
                        win_length=hp.win_length,
                        n_mel_channels=hp.n_mel_channels,
                        sampling_rate=hp.sampling_rate,
                        mel_fmin=hp.mel_fmin,
                        mel_fmax=hp.mel_fmax)

    wav_files = glob.glob(os.path.join(args.data_path, '**', '*.wav'), recursive=True)

    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        sr, wav = read_wav_np(wavpath)
        assert sr == hp.sampling_rate, \
            "sample rate mismatch. expected %d, got %d at %s" % \
            (hp.sampling_rate, sr, wavpath)
        
        if len(wav) < hp.segment_length + hp.pad_short:
            wav = np.pad(wav, (0, hp.segment_length + hp.pad_short - len(wav)), \
                    mode='constant', constant_values=0.0)

        wav = torch.from_numpy(wav).unsqueeze(0)
        mel = stft.mel_spectrogram(wav)

        melpath = os.path.join(args.out_path, os.path.basename(wavpath.replace('.wav', '.npy')))
        # amppath = os.path.join(args.out_path, os.path.basename(wavpath.replace('.wav', '_amp.npy')))
        np.save(melpath, mel.squeeze(0).transpose(0,1).numpy())
        # np.save(melpath, mel.squeeze(0).transpose(0,1).numpy())

def read_wav_np(path):
    sr, wav = read(path)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return sr, wav

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py')
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help="root directory of wav files")
    parser.add_argument('-o', '--out_path', type=str, default=None,
                        help="save directory of mel files")
    args = parser.parse_args()
    if args.out_path is None:
        args.out_path = args.data_path
    hp.configure(args.hp_file)

    main(hp, args)

