import argparse
from utils import hparams as hp

parser = argparse.ArgumentParser()
parser.add_argument('--hp_file', type=str, default='hparams.py')
parser.add_argument('--parameter', type=str, default='save_dir')
args = parser.parse_args()
hp_file = args.hp_file
parameter = args.parameter

hp.configure(hp_file)

assert hasattr(hp, parameter), f'{parameter} is not found in {hp_file}'

print(getattr(hp, parameter))
