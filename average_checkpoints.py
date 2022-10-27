# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np

def main():

    if args.start is None and args.end is None:
        print('average {} files from last modified model'.format(args.num))
        last = sorted(args.snapshots, key=os.path.getmtime)
        last = last[-args.num:]
    elif args.start is not None and args.end is not None:
        last = []
        dirname = os.path.dirname(args.snapshots[0])
        for epoch in range(args.start, args.end+1):
            last.append(os.path.join(dirname, 'network.epoch{}'.format(epoch)))
        
    print("average over", last)
    avg = None
    if args.num is None:
        args.num = args.end - args.start + 1

    if args.backend == 'pytorch':
        import torch
        # sum
        for path in last:
            print(path)
            #states = torch.load(path, map_location=torch.device("cpu"))["model"]
            states = torch.load(path, map_location=torch.device("cpu"))
            if avg is None:
                avg = states
            else:
                for k in avg.keys():
                    avg[k] += states[k]

        # average
        for k in avg.keys():
            if avg[k] is not None:
                avg[k] = torch.div(avg[k], args.num)

        torch.save(avg, args.out)
        print('{} saved.'.format(args.out))

    else:
        raise ValueError('Incorrect type of backend')


def get_parser():
    parser = argparse.ArgumentParser(description='average models from snapshot')
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=None, type=int)
    parser.add_argument("--start", default=None, type=int)
    parser.add_argument("--end", default=None, type=int)
    parser.add_argument("--backend", default='pytorch', type=str)
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    main()
