import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_txt')
    parser.add_argument('--save_dir')
    parser.add_argument('--ignore_labels', type=str, default=None)
    args = parser.parse_args()
    input_txt = args.input_txt
    save_dir = args.save_dir
    ignore_labels = args.ignore_labels

    ignore_labels_list = []
    if ignore_labels is not None:
        for ignore_label in ignore_labels.strip().split(','):
            ignore_labels_list.append(ignore_label)

    results_dict = {}
    with open(input_txt) as f:
        for line in f:
            line = line.strip()
            for word in line.split(' '):
                if not word in ignore_labels_list:
                    results_dict[word] = 1

    results_sorted = sorted(results_dict.items(), key=lambda x:x[0])
    print('0 <unk>')
    print('1 </s>')
    print('2 <s>')
    for i, (k, v) in enumerate(results_sorted):
        print(f'{i+3} {k}')
