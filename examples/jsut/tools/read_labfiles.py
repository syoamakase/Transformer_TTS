import argparse

def parse_label(lab_file, extract_lab, out_delimiter=','):
    with open(lab_file) as f:
        results = ''
        for line in f:
            start_time, end_time, context_label = line.split(' ')
            start_time = float(start_time)
            end_time = float(end_time)
            labels = context_label.split('/')
            phones = labels[0]
            phones = phones.split('-')[1].split('+')[0]
            if phones == 'xx':
                continue
            results += f'{phones}'
            # for l in labels[1:]:
            #     l_id = l.split(':')[0]
            #     if l_id in extract_lab:
            #         l = l.replace('|', ',')
            #         results += f'/{l}'
            results += ' '
    
    return results
    # 0 3125000 xx^xx-sil+m=i/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:02+xx_xx/E:xx_xx!xx_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx/G:3_3%0_xx_xx/H:xx_xx/I:xx-xx@xx+xx&xx-xx|xx+xx/J:5_23/K:1+5-23
    # 3125000 3525000 xx^sil-m+i=z/A:-2+1+3/B:xx-xx_xx/C:02_xx+xx/D:13+xx_xx/E:xx_xx!xx_xx-xx/F:3_3#0_xx@1_5|1_23/G:7_2%0_xx_1/H:xx_xx/I:5-23@1+1&1-5|1+23/J:xx_xx/K:1+5-2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lab_list')
    parser.add_argument('--extract_lab', nargs='*', default=['A', 'F'], help='label_id to extract')
    args = parser.parse_args()
    lab_list = args.lab_list
    extract_lab = args.extract_lab

    with open(lab_list) as f:
        for lab_file in f:
            lab_file = lab_file.strip()
            results = parse_label(lab_file, extract_lab)
            print(f'{lab_file}|{results}')