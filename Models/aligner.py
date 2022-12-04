
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Aligner(nn.Module):
    """The class for differentialble Duration Modeling

    """
    def __init__(self, hp, M):
        """
        M: max duration lengths
        """
        super().__init__()
        self.hp = hp
        self.d_model_e = hp.d_model_e
        kernel_size = 9
        self.M = M
        self.conv_layers = nn.ModuleList([
                                     nn.Conv1d(in_channels=self.d_model_e, out_channels=self.d_model_e, kernel_size=kernel_size, padding=8),
                                     nn.LayerNorm(self.d_model_e),
                                     nn.Dropout(0.1),
                                     nn.Conv1d(in_channels=self.d_model_e, out_channels=self.d_model_e, kernel_size=kernel_size, padding=8),
                                     nn.LayerNorm(self.d_model_e),
                                     nn.Dropout(0.1),
                                     nn.Conv1d(in_channels=self.d_model_e, out_channels=self.d_model_e, kernel_size=kernel_size, padding=8),
                                     nn.LayerNorm(self.d_model_e),
                                     nn.Dropout(0.1)])
        self.out = nn.Linear(self.d_model_e, self.M)

    def forward(self, encoded_feature):
        # encoded_feature (B, L, H)
        conv_out = self.conv_layers(encoded_feature.transpose(1, 2))

        outputs = self.out(conv_out.transpose(1, 2))
        noise = torch.randn(outputs).to(outputs.device)
        outputs = outputs + noise

        return torch.sigmoid(outputs)

    def convert_s(self, prob_l):
        pass