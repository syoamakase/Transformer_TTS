import argparse
import numpy as np
import matplotlib.pyplot as plt

#/n/work1/ueno/data/Erica/dataset_script/data/data/train/wav_16kHz/hoya_360-597_594.npy

#def plot_mel_and_alignment(mel, log_duration, text):
def plot_mel_and_alignment(mel, duration_rounded):
    from scipy import ndimage
    #duration_rounded = torch.clamp(torch.round(torch.exp(log_duration)-1), min=0)
    mel_len, mel_dim = mel.shape
    x = np.arange(0, mel_dim, 0.1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    mel = ndimage.rotate(mel, 90)
    ax1.imshow(mel)
    step = 0
    #for d, t in zip(duration_rounded[0], text[0]):
    for d in duration_rounded:
        ax1.plot([step+d, step+d], [0, mel_dim-1], "red", linestyle='dashed')
        #print(f'{step+d//2}')
        #ax1.text(step+d//2, 20, f'{t}')
        step += d
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    mel_name = args.filename
    ali_name = args.filename.replace('.npy', '_alignment.npy')
    mel = np.load(mel_name)
    ali = np.load(ali_name)
    plot_mel_and_alignment(mel, ali)
