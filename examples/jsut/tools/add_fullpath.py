import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labfile')
    parser.add_argument('--ext', type=str, default='npy')
    parser.add_argument('--save_dir', type=str, required=True)
    
    args = parser.parse_args()
    with open(args.labfile) as f:
        for line in f:
            line, text = line.strip().split('|', 1)
            line = line.replace('.lab', '.'+args.ext)

            relative_path = os.path.join(args.save_dir, line)
            print('{}|{}'.format(os.path.abspath(relative_path), text))            
