# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_lm(nn.Module):
    def __init__(self, hp_LM):
        super(Model_lm, self).__init__()

        self.embeddings_1 = nn.Embedding(hp_LM.num_classes, hp_LM.num_hidden_LM//2)
        self.embeddings_2 = nn.Embedding(hp_LM.num_classes, hp_LM.num_hidden_LM//2)
        self.lstm_1 = nn.LSTM(input_size=hp_LM.num_hidden_LM, hidden_size=hp_LM.num_hidden_LM, num_layers=4, dropout=0.2, batch_first=True)
        self.linear_1 = nn.Linear(hp_LM.num_hidden_LM, hp_LM.num_classes)
        #self.lstm_2 = nn.LSTM(input_size=hp_LM.num_hidden_LM, hidden_size=hp_LM.num_hidden_LM, num_layers=2, dropout=0.2, batch_first=True)
        self.linear_2 = nn.Linear(hp_LM.num_hidden_LM, hp_LM.num_classes)

    def forward(self, input1_, input2_):
        embeds1 = self.embeddings_1(input1_)
        embeds2 = self.embeddings_2(input2_)

        embeds = torch.cat((embeds1, embeds2), dim=2)
        lstm_out, (_, _) = self.lstm_1(embeds)
        #lstm_out2, (_, _) = self.lstm_2(embeds2)
        prediction1 = self.linear_1(lstm_out)
        prediction2 = self.linear_2(lstm_out)

        return prediction1, prediction2
