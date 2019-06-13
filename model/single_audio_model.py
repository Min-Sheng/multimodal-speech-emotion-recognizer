import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils.model_utils import get_glove

class SingleAudioModel(nn.Module):
    def __init__(self, input_size, prosody_size,
                 num_layers, 
                 hidden_dim, output_dim, 
                 dr, bidirectional):
        super(SingleAudioModel, self).__init__()
        
        self.input_size = input_size
        self.prosody_size = prosody_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dr = dr
        
        self.bi = bidirectional
        self.rnn = nn.GRU(self.input_size, self.hidden_dim, bias=True,
                           num_layers=self.num_layers, dropout=self.dr,
                           bidirectional=self.bi)
        
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.orthogonal_(self.rnn.__getattr__(p))
        
        self.dropout = nn.Dropout(p=self.dr)
        self.n_directions = 2 if self.bi else 1
        self.out = nn.Linear(self.n_directions * self.hidden_dim + self.prosody_size, self.output_dim)
        torch.nn.init.uniform_(self.out.weight, -0.25, 0.25)
        self.out.bias.data.fill_(0)

    def forward(self, input_seq, input_lengths, input_prosody):
        
        # Turn (batch_size, seq_len, hidden_dim) into (seq_len, batch_size, hidden_dim) for RNN
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq.transpose(0, 1), input_lengths, enforce_sorted=False)
        rnn_outputs, h_n = self.rnn(packed) # h_n: (num_layers * num_directions, batch, hidden_size)
        if self.bi:
            last_states_en = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_states_en = h_n[-1]
        last_states_en = self.dropout(last_states_en)
        encoded = torch.cat((last_states_en, input_prosody), dim=1)
        return self.out(encoded)