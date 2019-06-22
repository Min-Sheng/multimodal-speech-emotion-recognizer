import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils.model_utils import get_glove
from model.single_text_model import SingleTextModel
from model.single_audio_model import SingleAudioModel


class MultiModalModel(nn.Module):
    def __init__(self, dic_size, 
                 use_glove, 
                 num_layers_text, 
                 hidden_dim_text, 
                 embedding_dim_text, 
                 dr_text, bidirectional_text, 
                 embedding_train, 
                 input_size_audio, 
                 prosody_size, 
                 num_layers_audio, 
                 hidden_dim_audio, 
                 dr_audio, bidirectional_audio, 
                 output_dim):
        super(MultiModalModel, self).__init__()
        
        self.dic_size = dic_size
        self.use_glove = use_glove
        self.num_layers_text = num_layers_text
        self.hidden_dim_text = hidden_dim_text
        self.embedding_train = embedding_train

        if self.use_glove == 1:
            self.embedding_dim_text = 300
            
        else:
            self.embedding_dim_text = embedding_dim_text
        
        self.input_size_audio = input_size_audio
        self.prosody_size = prosody_size
        self.num_layers_audio = num_layers_audio
        self.hidden_dim_audio = hidden_dim_audio
        self.output_dim = output_dim
        self.dr_text = dr_text
        self.dr_audio = dr_audio
        self.bi_text = bidirectional_text
        self.bi_audio = bidirectional_audio

        
        self.single_text_encoder = SingleTextModel(dic_size = self.dic_size, 
                         use_glove = self.use_glove, 
                         num_layers = self.num_layers_text,
                         hidden_dim = self.hidden_dim_text, output_dim = self.output_dim, 
                         embedding_dim = self.embedding_dim_text, dr = self.dr_text, 
                         bidirectional = self.bi_text, embedding_train=self.embedding_train)
        
        self.n_directions_text = 2 if self.bi_text else 1
        text_out = nn.Linear(self.n_directions_text * self.hidden_dim_text, int(self.hidden_dim_text/2))

        self.single_text_encoder.out = text_out
        
        self.single_audio_encoder = SingleAudioModel(input_size = self.input_size_audio, prosody_size = self.prosody_size, 
                         num_layers = self.num_layers_audio, hidden_dim = self.hidden_dim_audio, output_dim = self.output_dim, 
                         dr = self.dr_audio, bidirectional = self.bi_audio)
        
        self.n_directions_audio = 2 if self.bi_audio else 1
        audio_out = nn.Linear(self.n_directions_audio * self.hidden_dim_audio + self.prosody_size, int(self.hidden_dim_audio/2))

        self.single_audio_encoder.out = audio_out
        self.out = nn.Linear(int((self.hidden_dim_text + self.hidden_dim_audio)/2), self.output_dim)
        torch.nn.init.uniform_(self.out.weight, -0.25, 0.25)
        self.out.bias.data.fill_(0)

    def forward(self, input_seq_text, input_lengths_text, input_seq_audio, input_lengths_audio, input_prosody):
        
        out_text = self.single_text_encoder(input_seq_text, input_lengths_text)
        out_audio = self.single_audio_encoder(input_seq_audio, input_lengths_audio, input_prosody)
        final = torch.cat((out_text, out_audio), dim=1)

        return self.out(final)