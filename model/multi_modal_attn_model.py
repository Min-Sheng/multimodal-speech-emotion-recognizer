import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils.model_utils import get_glove
from model.single_text_model import SingleTextModel
from model.single_audio_model import SingleAudioModel

class Attn(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attn, self).__init__()

        self.dimensions = dimensions
        self.attention_type = attention_type

        if self.attention_type == 'general':
            self.linear_in = nn.Linear (self.dimensions, self.dimensions, bias=False)
        
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        
    def forward(self, query, context):
        
        # query [batch(128), dimensions(1024)] -> query [batch(128), output_len(1), dimensions(1024)]
        # context [query_len, batch(128), dimensions(1024)] -> context [batch(128), query_len, dimensions(1024)]
        query = query.unsqueeze(1)
        context = context.transpose(0, 1)
        
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        
        if self.attention_type == "general":
            # query [batch(128), output_len(1), dimensions(1024)]
            # -> query [batch(128) * output_len(1), dimensions(1024)]
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            
            # query [batch(128) * output_len(1), dimensions(1024)]
            # -> query [batch(128), output_len(1), dimensions(1024)]
            query = query.view(batch_size, output_len, dimensions)
            
        # query [batch(128) * output_len(1), dimensions(1024)] * context [batch(128), query_len, dimensions(1024)] 
        # -> attention_scores [batch_size, output_len, query_len]
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        
        # Compute weights across every context sequence
        # attention_scores [batch_size(128), output_len(1), query_len]
        # -> attention_scores [batch_size(128) * output_len(1), query_len]
        # -> attention_weights [batch_size(128), query_len]
        # -> attention_weights [batch_size(128), output_len(1), query_len]
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        
        # attention_weights [batch_size(128), output_len(1), query_len] * context [batch(128), query_len, dimensions(1024)]
        # -> attened [batch_size(128), output_len(1), dimensions(1024))]
        attened = torch.bmm(attention_weights, context)

        # -> combined [batch_size(128) * output_len(1), 2*dimensions(1024*2))]
        combined = torch.cat((attened, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on combined
        # -> output [batch_size(128), output_len(1), dimensions(1024)]
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)

        return output, attention_weights
        
class MultiModalAttnModel(nn.Module):
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
                 output_dim, attention_type = 'general',
                 t2a_attn = True):
        super(MultiModalAttnModel, self).__init__()
        
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
        self.attention_type = attention_type
        self.t2a_attn = t2a_attn
        
        self.single_text_encoder = SingleTextModel(dic_size = self.dic_size, 
                         use_glove = self.use_glove, 
                         num_layers = self.num_layers_text,
                         hidden_dim = self.hidden_dim_text, output_dim = self.output_dim, 
                         embedding_dim = self.embedding_dim_text, dr = self.dr_text, 
                         bidirectional = self.bi_text, out_en=True, embedding_train=self.embedding_train)
        
        self.n_directions_text = 2 if self.bi_text else 1
        # text_out [self.n_directions_text * self.hidden_dim_text]
        self.single_text_encoder.out = nn.Identity()
        
        if self.t2a_attn:
            self.single_audio_encoder = SingleAudioModel(input_size = self.input_size_audio, prosody_size = self.prosody_size, 
                             num_layers = self.num_layers_audio, hidden_dim = self.hidden_dim_audio, output_dim = self.output_dim, 
                             dr = self.dr_audio, bidirectional = self.bi_audio, out_en=True)
        else:
            self.single_audio_encoder = SingleAudioModel(input_size = self.input_size_audio, prosody_size = self.prosody_size, 
                             num_layers = self.num_layers_audio, hidden_dim = self.hidden_dim_audio, output_dim = self.output_dim, 
                             dr = self.dr_audio, bidirectional = self.bi_audio)
        
        self.n_directions_audio = 2 if self.bi_audio else 1
        # audio_out [self.n_directions_audio * self.hidden_dim_audio + self.prosody_size]
        self.single_audio_encoder.out = nn.Identity()
        
        self.fc_audio = nn.Linear(self.n_directions_audio * self.hidden_dim_audio + self.prosody_size, self.n_directions_text * self.hidden_dim_text)
        torch.nn.init.uniform_(self.fc_audio.weight, -0.25, 0.25)
        self.fc_audio.bias.data.fill_(0)
        
        if self.t2a_attn:
            self.fc_text = nn.Linear(self.n_directions_text * self.hidden_dim_text, self.n_directions_audio * self.hidden_dim_audio)
            torch.nn.init.uniform_(self.fc_text.weight, -0.25, 0.25)
            self.fc_text.bias.data.fill_(0)
            
        if attention_type != 'none':
            self.attn_text = Attn(self.n_directions_text * self.hidden_dim_text, attention_type)
            if self.t2a_attn:
                self.attn_audio = Attn(self.n_directions_audio * self.hidden_dim_audio, attention_type)
        
        if self.t2a_attn:
            self.out = nn.Linear((self.n_directions_text * self.hidden_dim_text) * 2 + self.prosody_size, self.output_dim)
            torch.nn.init.uniform_(self.out.weight, -0.25, 0.25)
            self.out.bias.data.fill_(0)
        else:
            self.out = nn.Linear((self.n_directions_text * self.hidden_dim_text) * 2, self.output_dim)
            torch.nn.init.uniform_(self.out.weight, -0.25, 0.25)
            self.out.bias.data.fill_(0)

    def forward(self, input_seq_text, input_lengths_text, input_seq_audio, input_lengths_audio, input_prosody):
        
        text_encoded, text_out_en = self.single_text_encoder(input_seq_text, input_lengths_text)
        
        if self.t2a_attn:
            audio_encoded, audio_out_en = self.single_audio_encoder(input_seq_audio, input_lengths_audio, input_prosody)
            audio_encoded = self.fc_audio(audio_encoded)
            text_encoded = self.fc_text(text_encoded)

            attn_text_encoded, attn_text_weights = self.attn_text(audio_encoded, text_out_en)
            attn_audio_encoded, attn_audio_weights = self.attn_audio(text_encoded, audio_out_en)
            
            final_encoded = torch.cat((attn_text_encoded.squeeze(), attn_audio_encoded.squeeze()), dim=1)
            final_encoded = torch.cat((final_encoded, input_prosody), dim=1)
            
        else:
            audio_encoded = self.single_audio_encoder(input_seq_audio, input_lengths_audio, input_prosody)
            audio_encoded = self.fc_audio(audio_encoded)
            
            attn_text_encoded, attn_text_weights = self.attn_text(audio_encoded, text_out_en)
            
            final_encoded = torch.cat((attn_text_encoded.squeeze(), audio_encoded), dim=1)
        return self.out(final_encoded)