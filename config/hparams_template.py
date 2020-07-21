## CONFIG

train_script = 'examples/jsut/data/train/script_24000/train_id_sort_xlen.txt'
test_script = 'examples/jsut/data/dev/script_24000/dev_id.txt' 
spm_model =  None
mean_file = 'examples/jsut/data/train/script_24000/mean.npy'
var_file = 'examples/jsut/data/train/script_24000/var.npy'
vocab_size = 44

save_dir = 'checkpoints.Transformer.JSUT'
log_dir = 'checkpoints.Transformer.JSUT'

# if not exist, genereate automatically
lengths_file = 'examples/jsut/data/train/script_24000/engths.npy'
max_seqlen = 100000
batch_size = None # None or 120
warmup_step = 4000
warmup_factor = 1.0

## General config
mel_dim = 80

## Training hyper-parameters
# batch_size = 64
reduction_rate = 5
max_epoch = 1000
average_start = 1000 - 9
save_per_epoch = 100
save_attention_per_step = 1000

# If you want to continue training from the middle, please specify
loaded_epoch = None
loaded_dir = None 

# setting of Transformer
d_model_encoder = 384
n_layer_encoder = 6
n_head_encoder = 4
d_model_decoder = 384
n_layer_decoder = 6
n_head_decoder = 4
ff_conv_kernel_size_encoder = 5
ff_conv_kernel_size_decoder = 1
concat_after_encoder = False
concat_after_decoder = False

dropout = 0.1
prenet_dropout_rate = 0.5

# in the future remove?
postnet_pred = True
optimizer = 'Noam'
comment = ''

is_multi_speaker = False
if is_multi_speaker:
    spk_emb_type = 'x_vector' #'speaker_id' or 'x_vector'
