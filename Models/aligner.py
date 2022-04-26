
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Aligner(nn.Module):
    """The class for differentialble Duration Modeling

    """
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        kernel_size = 9
        self.conv_layers = nn.ModuleList([
                                     nn.Conv1d(in_channels=self.d_model_e, out_channels=self.d_model_e, kernel_size=kernel_size, padding=8)])
