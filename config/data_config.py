################################
#     Data             
################################
DATA_PATH = '../multimodal-speech-emotion/data/processed/IEMOCAP/four_category/audio_woZ_set_all/'

DATA_TRAIN_MFCC            = 'train_audio_mfcc.npy'
DATA_TRAIN_MFCC_SEQN  = 'train_seqN.npy'
DATA_TRAIN_PROSODY      = 'train_audio_prosody.npy'
#DATA_TRAIN_PROSODY      = 'train_audio_emobase2010.npy'   # easy emobase2010 setting
DATA_TRAIN_LABEL           = 'train_label.npy'
DATA_TRAIN_TRANS          = 'train_nlp_trans.npy'


DATA_DEV_MFCC              = 'dev_audio_mfcc.npy'
DATA_DEV_MFCC_SEQN    = 'dev_seqN.npy'
DATA_DEV_PROSODY        = 'dev_audio_prosody.npy'
#DATA_DEV_PROSODY        = 'dev_audio_emobase2010.npy'   # easy emobase2010 setting
DATA_DEV_LABEL             = 'dev_label.npy'
DATA_DEV_TRANS            = 'dev_nlp_trans.npy'


DATA_TEST_MFCC            = 'test_audio_mfcc.npy'
DATA_TEST_MFCC_SEQN  = 'test_seqN.npy'
DATA_TEST_PROSODY      = 'test_audio_prosody.npy'
#DATA_TEST_PROSODY     = 'test_audio_emobase2010.npy'   # easy emobase2010 setting
DATA_TEST_LABEL           = 'test_label.npy'
DATA_TEST_TRANS          = 'test_nlp_trans.npy'


DIC                               = 'dic.pkl'
GLOVE                              = 'W_embedding.npy'
