import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_txt')
    parser.add_argument('--vocab_id', required=True)
    parser.add_argument('--ignore_labels', type=str, default=None)
    args = parser.parse_args()
    input_txt = args.train_txt
    vocab_id = args.vocab_id
    ignore_labels = args.ignore_labels

    ignore_labels_list = []
    if ignore_labels is not None:
        for ignore_label in ignore_labels.strip().split(','):
            ignore_labels_list.append(ignore_label)

    vocab_dict = {}
    with open(vocab_id) as f:
        for line in f:
            v_id, vocab = line.strip().split(' ')
            vocab_dict[vocab] = v_id

    with open(input_txt) as f:
        for line in f:
            file_id, line = line.strip().split('|')
            print(file_id, end='|')
            for word in line.split(' '):
                if not word in ignore_labels_list:
                    print(vocab_dict[word], end=' ')
            print('1')