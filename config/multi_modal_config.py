################################
#     Training             
################################
MODEL_NAME = 'multimodalEncoder'
CAL_ACCURACY_FROM = 0
MAX_EARLY_STOP_COUNT = -1
EPOCH_PER_VALID_FREQ = 0.3
GPU = 0
N_STEPS = 6000
LR = 0.001
BATCH_SIZE = 128
PRINT_EVERY = 100
WEIGHT_DECAY = 0.01 #0.001

################################
#     COMMON
################################
N_CATEGORY = 4

################################
#     NLP
################################
N_SEQ_MAX_NLP = 128
USE_GLOVE = True
DIM_WORD_EMBEDDING = 100   # when using glove it goes to 300 automatically
EMBEDDING_TRAIN = True           # True is better
N_LAYER_TEXT = 2
DROPOUT_RATE_TEXT = 0.3
HIDDEN_DIM_TEXT = 512 #200
ENCODER_SIZE_TEXT = 128
BIDIRECTIONAL_TEXT = True


################################
#     Audio
################################
N_AUDIO_MFCC = 39
N_AUDIO_PROSODY= 35
#N_AUDIO_PROSODY= 1582   # easy emobase2010 setting
N_LAYER_AUDIO = 2
DROPOUT_RATE_AUDIO = 0.7
HIDDEN_DIM_AUDIO = 512 #200
ENCODER_SIZE_AUDIO = 128
BIDIRECTIONAL_AUDIO = True
