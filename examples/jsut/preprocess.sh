#!/bin/bash

sampling_rate=24000
wav_dir='wav_24kHz'

# Please specify the data path of JSUT data
jsut_path='jsut_ver1.1'

# download data
git clone https://github.com/r9y9/jsut-lab

if [ ${sampling_rate} = 16000 ]; then
  hparams_path='hparams_16kHz.py'
else
  hparams_path='hparams_24kHz.py'
fi

mkdir -p data/train data/dev
mkdir -p data/train/${wav_dir}
mkdir -p data/dev/${wav_dir}
mkdir -p data/train/script_${sampling_rate}
mkdir -p data/dev/script_${sampling_rate}
mkdir -p tmp

echo -n "" > lablist
echo -n "" > wavlist
for file in basic5000 countersuffix26 loanword128 onomatopee300 precedent130 repeat500 travel1000 utparaphrase512 voiceactress100; do
   readlink -f jsut-lab/${file}/lab/* >> lablist
   readlink -f ${jsut_path}/${file}/wav/* >> wavlist  
done

python tools/read_labfiles.py lablist > data/train/train_all.txt
head -n 50 data/train/train_all.txt > data/dev/dev.txt
head -n 50 wavlist > data/dev/wavlist
tail -n+50 data/train/train_all.txt > data/train/train.txt
tail -n+50 wavlist > data/train/wavlist

python tools/make_sox.py data/train/wavlist --save_dir data/train/${wav_dir} --sampling_rate ${sampling_rate} | bash
python tools/make_sox.py data/dev/wavlist --save_dir data/dev/${wav_dir} --sampling_rate ${sampling_rate} | bash

python preprocess.py --hp_file config/${hparams_path} -d data/train/${wav_dir}
python preprocess.py --hp_file config/${hparams_path} -d data/dev/${wav_dir}

python tools/cut.py -d '|' -f 1- data/train/train_all.txt > data/train/input_sentencepiece.txt

python tools/make_vocab.py data/train/input_sentencepiece.txt --ignore_labels "sil" > data/train/vocab.id
python tools/apply_vocab.py data/train/train.txt --vocab_id data/train/vocab.id --ignore_labels "sil" > tmp/train_id.txt
python tools/apply_vocab.py data/dev/dev.txt --vocab_id data/train/vocab.id --ignore_labels "sil" > tmp/dev_id.txt

python tools/cut.py -d '/' -f -1 tmp/train_id.txt > tmp/train_id.2.txt
python tools/cut.py -d '/' -f -1 tmp/dev_id.txt > tmp/dev_id.2.txt

python tools/add_fullpath.py tmp/train_id.2.txt --save_dir data/train/${wav_dir} --ext npy > data/train/script_${sampling_rate}/train_id.txt
python tools/add_fullpath.py tmp/dev_id.2.txt --save_dir data/dev/${wav_dir} --ext npy > data/dev/script_${sampling_rate}/dev_id.txt

python tools/sort_by_xlen.py -S data/train/script_${sampling_rate}/train_id.txt > data/train/script_${sampling_rate}/train_id_sort_xlen.txt
python tools/feature_normalize.py -S data/train/script_${sampling_rate}/train_id.txt --ext npy --save_dir data/train/script_${sampling_rate}/

echo "complete!"
echo "See the results to check file path."
train_path=`readlink -f data/train/script_${sampling_rate}/train_id_sort_xlen.txt`
dev_path=`readlink -f data/dev/script_${sampling_rate}/dev_id.txt`
mean_file=`readlink -f data/train/script_${sampling_rate}/mean.npy`
var_file=`readlink -f data/train/script_${sampling_rate}/var.npy`
vocab_size=`wc -l < data/train/vocab.id`
echo "train_script = ${train_path}" > results
echo "test_script = ${dev_path}" >> results
echo "mean_file = ${mean_file}" >> results
echo "var_file = ${var_file}" >> results
echo "vocab_size = ${vocab_size}" >> results