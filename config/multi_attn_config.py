################################
#     Training             
################################
CAL_ACCURACY_FROM = 0
MAX_EARLY_STOP_COUNT = 100
EPOCH_PER_VALID_FREQ = 0.3

################################
#     Audio
################################
N_CATEGORY = 4
N_AUDIO_MFCC = 39
N_AUDIO_PROSODY= 35
#N_AUDIO_PROSODY= 1582   # easy emobase2010 setting
N_SEQ_MAX = 750                 # max 1,000 (MSP case only)


################################
#     NLP
################################
N_SEQ_MAX_NLP = 128
DIM_WORD_EMBEDDING = 100   # when using glove it goes to 300 automatically
EMBEDDING_TRAIN = True           # True is better
