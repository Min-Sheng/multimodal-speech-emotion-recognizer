################################
#     Training             
################################
MODEL_NAME = 'a2eEncoder'
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
#     Audio
################################
N_CATEGORY = 4
N_AUDIO_MFCC = 39
N_AUDIO_PROSODY= 35
#N_AUDIO_PROSODY= 1582   # easy emobase2010 setting
N_LAYER = 2
DROPOUT_RATE = 0.7
HIDDEN_DIM = 512 #200
ENCODER_SIZE = 128
BIDIRECTIONAL = True