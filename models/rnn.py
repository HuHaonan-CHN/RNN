import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, x, x_lens):
        # x [batch_size, sample_num, sequence_len, enc_in]
        B, N_S, L, I_D = x.shape
        x = x.reshape(B * N_S, L, I_D)  # x_in [B*N_S,L,I_D]

        x_out, _ = self.rnn(x)  # x_out [B*N_S,L,H_D]

        out = torch.zeros((B * N_S, self.hidden_size)).to(self.device)

        for index, this_len in enumerate(x_lens):
            out[index] = x_out[index][this_len - 1]

        out = out.reshape(B, N_S, self.hidden_size)

        return out

