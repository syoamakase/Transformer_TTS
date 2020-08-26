# Transformer_TTS

This repository is Text-to-Speech (TTS) project based on Transformer.

## train

`python train.py --hp_file config/hparams_template.py`

If you want to check a loss curve, `tensorboard --logdir <save_dir>/logs`

### hparams.py

`hparams.py` is a file to control hyper parameters.

When you use your own dataset, you must adjust`train_script`, `test_script`, `mean_file`, `var_file`, and `vocab_size`.

## test

When you generate a speech, please take an average.
`python utils/average_checkpoints.py --backend pytorch --snapshots <save directory>/network.epoch* --out <save directory>/network.average_epoch991-epoch1000 --start 991 --end 1000`


`python test.py --load_name <model path>`

# TODO
- add FastSpeech2
