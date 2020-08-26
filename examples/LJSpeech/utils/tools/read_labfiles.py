import argparse
from g2p_en import G2p

def parse_label(meta_data, g2p):
    with open(meta_data) as f:
        for line in f:
            file_id, _, text = line.strip().split('|')
            phones = g2p(text)

            texts = ''
            for p in phones:
                if p == ' ':
                    texts += '<space> '
                else:
                    texts += p + ' '

            print(f'{file_id}|{texts}') 
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_data')
    args = parser.parse_args()
    meta_data = args.meta_data

    g2p = G2p()
    results = parse_label(meta_data, g2p)
