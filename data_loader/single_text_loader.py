"""
what    : process data, generate batch
"""

import torch
import numpy as np
import pickle
import random

from config.data_config import *
from config.single_text_config import *

class SingleTextLoader:
    
    def __init__(self, data_path = DATA_PATH):
        
        self.data_path = data_path
        
        # load data
        self.train_set = self.load_data(DATA_TRAIN_TRANS, DATA_TRAIN_LABEL)
        self.dev_set  = self.load_data(DATA_DEV_TRANS, DATA_DEV_LABEL)
        self.test_set  = self.load_data(DATA_TEST_TRANS, DATA_TEST_LABEL)
        
        self.dic_size = 0
        with open( data_path + DIC , "rb") as f:
            self.dic_size = len( pickle.load(f) )
    
        
    def load_data(self, text_trans, label):
     
        print('load data : ' + text_trans + ' ' + label)
        output_set = []

        tmp_text_trans          = np.load(self.data_path + text_trans)
        tmp_label               = np.load(self.data_path + label)

        for i in range( len(tmp_label) ) :
            output_set.append( [tmp_text_trans[i], tmp_label[i]] )
        print('[completed] load data')
        
        return output_set
    
    """
        inputs: 
            data            : data to be processed (train/dev/test)
            batch_size      : mini-batch size
            encoder_size    : max encoder time step
            
            is_test         : True, inference stage (ordered input)  ( default : False )
            start_index     : start index of mini-batch

        return:
            encoder_input   : [batch, time_step(==encoder_size)]
            encoder_seq     : [batch] - valid word sequence
            labels               : [batch, category] - category is one-hot vector
    """
    def get_batch(self, data, batch_size = BATCH_SIZE, encoder_size = ENCODER_SIZE, is_test=False, start_index=0):

        encoder_inputs, encoder_seq, labels = [], [], []
        index = start_index
        
        # Get a random batch of encoder and encoderR inputs from data,
        # pad them if needed

        for _ in range(batch_size):

            if is_test is False:
                # train case -  random sampling
                trans, label = random.choice(data)
                
            else:
                # dev, test case = ordered data
                if index >= len(data):
                    trans, label = data[0]  # won't be evaluated
                    index += 1
                else: 
                    trans, label = data[index]
                    index += 1
            
            tmp_index = np.where( trans == 0 )[0]   # find the pad index
            if ( len(tmp_index) > 0 ) :             # pad exists
                seqN =  np.min((tmp_index[0],encoder_size))
            else :                                  # no-pad
                seqN = encoder_size
                
            encoder_inputs.append( trans[:encoder_size] )
            encoder_seq.append( seqN )
            
            tmp_label = np.zeros( N_CATEGORY, dtype=np.float )
            tmp_label[label] = 1
            labels.append( tmp_label )
        
        #encoder_inputs, encoder_seq, labels = zip(*sorted(zip(encoder_inputs, encoder_seq, labels), key=lambda tup: tup[1], reverse=True))
        
        return torch.LongTensor(encoder_inputs), torch.LongTensor(encoder_seq), torch.FloatTensor(np.reshape(labels, (batch_size,N_CATEGORY)))