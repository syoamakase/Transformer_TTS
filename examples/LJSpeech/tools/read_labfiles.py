import argparse
import re
from g2p_en import G2p

from text import text_to_sequence, sequence_to_text

def parse_label(meta_data, g2p):
    with open(meta_data) as f:
        for line in f:
            file_id, _, texts = line.strip().split('|')
            
            phone = g2p(texts)

            # texts = ''
            # for p in phones:
            #     if p == ' ':
            #         texts += '<space> '
            #     else:
            #         texts += p + ' '
            phone = list(filter(lambda p: p != ' ', phone))
            phone = '{'+ '}{'.join(phone) + '}'
            phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
            phone = phone.replace('}{', ' ')
            texts = phone
            #texts = ' '.join([str(i) for i in text_to_sequence(phone, ['english_cleaners'])])

            print(f'{file_id}|{texts}') 
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_data')
    args = parser.parse_args()
    meta_data = args.meta_data

    g2p = G2p()
    results = parse_label(meta_data, g2p)
