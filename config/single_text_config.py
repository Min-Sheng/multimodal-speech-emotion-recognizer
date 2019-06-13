################################
#     Training             
################################
MODEL_NAME = 't2eEncoder',
CAL_ACCURACY_FROM = 0
MAX_EARLY_STOP_COUNT = 100
EPOCH_PER_VALID_FREQ = 0.3
GPU = 0
N_STEPS = 100000
LR = 0.001
BATCH_SIZE = 128
PRINT_EVERY = 100
WEIGHT_DECAY = 0.001

################################
#     NLP
################################
N_CATEGORY = 4
N_SEQ_MAX_NLP = 128
USE_GLOVE = True
DIM_WORD_EMBEDDING = 100   # when using glove it goes to 300 automatically
EMBEDDING_TRAIN = True           # True is better
N_LAYER = 2
DROPOUT_RATE = 0.3
HIDDEN_DIM = 200
ENCODER_SIZE = 128
BIDIRECTIONAL = True
