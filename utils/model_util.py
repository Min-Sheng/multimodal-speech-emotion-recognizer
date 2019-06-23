import numpy as np
from config.data_config import *

def get_glove(data_path = DATA_PATH):
    return np.load(data_path + GLOVE)