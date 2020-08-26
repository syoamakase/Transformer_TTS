import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wavlist')
    parser.add_argument('--save_dir')
    parser.add_argument('--sampling_rate', type=int, default=24000)
    args = parser.parse_args()
    wavlist = args.wavlist
    save_dir = args.save_dir
    sampling_rate = args.sampling_rate

    with open(wavlist) as f:
        for wf in f:
            wf = wf.strip()
            basename = os.path.basename(wf)
            save_name = os.path.join(save_dir, basename)
            print(f'sox -G {wf} {save_name} rate -v {sampling_rate}')
