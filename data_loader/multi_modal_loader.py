import torch
import numpy as np
import pickle
import random

from config.data_config import *
from config.multi_modal_config import *

class MultiModalLoader:
    
    def __init__(self, data_path = DATA_PATH):
        
        self.data_path = data_path
        
        # load data
        self.train_set = self.load_data(DATA_TRAIN_MFCC, DATA_TRAIN_MFCC_SEQN, DATA_TRAIN_PROSODY, DATA_TRAIN_TRANS, DATA_TRAIN_LABEL)
        self.dev_set  = self.load_data(DATA_DEV_MFCC, DATA_DEV_MFCC_SEQN, DATA_DEV_PROSODY, DATA_DEV_TRANS, DATA_DEV_LABEL)
        self.test_set  = self.load_data(DATA_TEST_MFCC, DATA_TEST_MFCC_SEQN, DATA_TEST_PROSODY, DATA_TEST_TRANS, DATA_TEST_LABEL)
       
        self.dic_size = 0
        with open( data_path + DIC , "rb") as f:
            self.dic_size = len( pickle.load(f) )
        
    def load_data(self, audio_mfcc, mfcc_seqN, audio_prosody, text_trans, label):
     
        print('load data : ' + audio_mfcc + ' ' +  mfcc_seqN + ' ' + audio_prosody + ' ' + text_trans + ' ' + label)
        output_set = []

        # audio
        tmp_audio_mfcc          = np.load(self.data_path + audio_mfcc)
        tmp_mfcc_seqN          = np.load(self.data_path + mfcc_seqN)
        tmp_audio_prosody     = np.load(self.data_path + audio_prosody)
        tmp_label                   = np.load(self.data_path + label)
        
        # text
        tmp_text_trans          = np.load(self.data_path + text_trans)

        for i in range( len(tmp_label) ) :
            output_set.append( [tmp_audio_mfcc[i], tmp_mfcc_seqN[i], tmp_audio_prosody[i], tmp_text_trans[i], tmp_label[i]] )
        print('[completed] load data')
        
        return output_set

    """
        inputs: 
            batch_size              : mini-batch size
            data                       : data to be processed (train/dev/test)
            
            encoder_size_audio : max encoder time step
            encoder_size_text    : max encoder time step
            
            is_test           : True, inference stage (ordered input)  ( default : False )
            start_index     : start index of mini-batch

        return:
            encoder_input_audio   : [batch, time_step(==encoder_size), mfcc_dim]
            encoder_seq_audio     : [batch] - valid mfcc step
            encoder_prosody         : [batch, prosody_dim] 
            
            encoder_input_text     : [batch, time_step(==encoder_size)]
            encoder_seq_text       : [batch] - valid word sequence
            
            labels                         : [batch, category] - category is one-hot vector
    """
    def get_batch(self, batch_size, data, encoder_size_audio, encoder_size_text, is_test=False, start_index=0):

        encoder_inputs_audio, encoder_seq_audio, encoder_prosody, labels = [], [], [], []
        encoder_inputs_text, encoder_seq_text = [], []
        index = start_index
        
        # Get a random batch of encoder and encoderR inputs from data,
        # pad them if needed

        for _ in range(batch_size):

            if is_test is False:
                # train case -  random sampling
                mfcc, seqN_audio, prosody, trans, label = random.choice( data )
                
            else:
                # dev, test case = ordered data
                if index >= len( data ):
                    mfcc, seqN_audio, prosody, trans, label = data[0]            # won't be evaluated                    
                    index += 1
                else: 
                    mfcc, seqN_audio, prosody, trans, label= data[index]
                    index += 1

            # mix mfcc and prosody feature (not effective)
            # mfcc [seq_max x mfcc_dim],  prosody  [prosody_dim,]
            # audio_mix = [seq_max, mfcc+prosody]
            # prosody_tile = np.reshape(  np.tile(prosody,(N_SEQ_MAX)), (N_SEQ_MAX, N_AUDIO_PROSODY) )
            # audio_mix = np.concatenate( (mfcc,prosody_tile), axis=1 )

            audio_mix = mfcc
            encoder_inputs_audio.append( audio_mix[:encoder_size_audio] )
            encoder_seq_audio.append( np.min((seqN_audio,encoder_size_audio)) )
            encoder_prosody.append( prosody )
            
            
            seqN_text = 0
            tmp_index = np.where( trans == 0 )[0]                               # find the pad index
            
            if ( len(tmp_index) > 0 ) :                                                   # pad exists
                seqN_text =  np.min((tmp_index[0],encoder_size_text))
            else :                                                                               # no-pad
                seqN_text = encoder_size_text

            encoder_inputs_text.append( trans[:encoder_size_text] )
            encoder_seq_text.append( seqN_text )

            
            tmp_label = np.zeros( N_CATEGORY, dtype=np.float )
            tmp_label[label] = 1
            labels.append( tmp_label )
        
        return torch.LongTensor(encoder_inputs_text), torch.LongTensor(encoder_seq_text), torch.FloatTensor(encoder_inputs_audio), torch.LongTensor(encoder_seq_audio), torch.FloatTensor(encoder_prosody), torch.FloatTensor(np.reshape(labels, (batch_size,N_CATEGORY)))