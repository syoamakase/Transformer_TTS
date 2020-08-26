import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-f', type=str)
parser.add_argument('-d','--delimiter', type=str, default=' ')
args = parser.parse_args()

filename = args.filename
args_f = args.f

if '-' in args_f and args_f != '-1':
    is_continuous = True
    if len(args_f.split('-')) == 2:
        start_f = int(args_f.strip().split('-')[0])
        if args_f.strip().split('-')[1] == '':
            end_f = None
        else:
            end_f = int(args_f.strip().split('-')[1])
else:
    is_continuous = False
    start_f = int(args_f)

delimiter = args.delimiter

with open(filename) as f:
    for line in f:
        line = line.strip()
        if is_continuous:
            if end_f is None:
                sp_line = line.split(delimiter)[start_f:]
            else:
                sp_line = line.split(delimiter)[start_f:end_f]
            print(delimiter.join(sp_line))
        else:
            sp_line = line.split(delimiter)[start_f]
            print(sp_line)
