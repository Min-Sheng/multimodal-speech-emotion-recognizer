import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils.model_utils import get_glove

class SingleTextModel(nn.Module):
    def __init__(self, dic_size, 
                 use_glove, 
                 num_layers, 
                 hidden_dim, output_dim, 
                 embedding_dim, dr, 
                 bidirectional, out_en=False,
                 embedding_train=True):
        super(SingleTextModel, self).__init__()
        
        self.dic_size = dic_size
        self.use_glove = use_glove
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dr = dr
        self.out_en = out_en
        self.embedding_train = embedding_train
        
        if self.use_glove == 1:
            self.embedding_dim = 300
            self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(get_glove()).float(), freeze=(~self.embedding_train))
            
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(self.dic_size, self.embedding_dim)
        
        self.bi = bidirectional
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, bias=True,
                           num_layers=self.num_layers, dropout=self.dr,
                           bidirectional=self.bi)
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.orthogonal_(self.rnn.__getattr__(p))
        self.dropout = nn.Dropout(p=self.dr)
        self.n_directions = 2 if self.bi else 1
        self.out = nn.Linear(self.n_directions * self.hidden_dim, self.output_dim)
        torch.nn.init.uniform_(self.out.weight, -0.25, 0.25)
        self.out.bias.data.fill_(0)

    def forward(self, input_seq, input_lengths):
        
        # Turn (batch_size, seq_len) into (seq_len, batch_size) for RNN
        embedded = self.embedding(input_seq.t())
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        rnn_outputs, h_n = self.rnn(packed) # h_n: (num_layers * num_directions, batch, hidden_size)
        if self.bi:
            last_states_en = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_states_en = h_n[-1]
        last_states_en = self.dropout(last_states_en)
        
        if self.out_en:
            rnn_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outputs)
            return self.out(last_states_en), rnn_outputs
        
        else:
            return self.out(last_states_en)