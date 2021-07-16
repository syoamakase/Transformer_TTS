import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.modules import MultiHeadAttention

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class StyleEmbedding(nn.Module):
    def __init__(self, hp):
        super(StyleEmbedding, self).__init__()
        self.reference_encoder = ReferenceEncoder(hp)
        self.style_token_layer = StyleTokenLayer(hp)

    def forward(self, mel, mel_mask):
        reference_encoder_output = self.reference_encoder(mel, mel_mask)
        style_embedding, attn = self.style_token_layer(reference_encoder_output)

        return style_embedding

class ReferenceEncoder(nn.Module):
    def __init__(self, hp):
        super(ReferenceEncoder, self).__init__()

        mel_dim = hp.mel_dim

        # six 2-D convolutional layers with 3x3, 2x2 stride
        cnn_dim = (32, 32, 64, 64, 128, 128)

        padding = (3 - 1) // 2
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, cnn_dim[0], (3, 3), (2, 2), padding=padding, bias=False)])
        self.norm = nn.ModuleList([nn.BatchNorm2d(cnn_dim[0])])
        out_dim = (mel_dim+1) // 2

        for i in range(1, len(cnn_dim)):
            self.conv_layers.extend([nn.Conv2d(cnn_dim[i-1], cnn_dim[i], (3, 3), (2, 2), padding=padding, bias=False)])
            self.norm.extend([nn.BatchNorm2d(cnn_dim[i])])
            out_dim = (out_dim+1)//2

        self.gru = nn.GRU(128*out_dim, 128, 1, batch_first=True)

    def forward(self, x, mask):
        x = x.unsqueeze(1)
        for conv, norm in zip(self.conv_layers, self.norm):
            x = torch.relu(norm(conv(x)))

        B, C, T, H = x.shape
        x, _ = self.gru(x.reshape(B, T, H*C))
        x = x[:, -1, :]
        return x

class StyleTokenLayer(nn.Module):
    def __init__(self, hp):
        super(StyleTokenLayer, self).__init__()

        n_tokens = 10

        self.embeddings = nn.Parameter(torch.zeros((n_tokens, 384)))
        torch.nn.init.xavier_uniform_(self.embeddings)
        self.attention = MultiHeadAttention(heads=4, q_dim=128, k_dim=384, v_dim=384, d_model=384, dropout=0.1)

    def forward(self, reference_encoder_output):
        ## attention, each token embedding is 256-D
        # need tanh embedding?
        batch_size = reference_encoder_output.shape[0]
        embeddings = torch.tanh(self.embeddings).unsqueeze(0).expand(batch_size, -1, -1)
        x, attn = self.attention(reference_encoder_output.unsqueeze(1), embeddings, embeddings)
        #import matplotlib.pyplot as plt
        #import pdb; pdb.set_trace()

        return x, attn


