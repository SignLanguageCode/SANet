# Data config
import math

CONNECTION = [[0, 1], [1, 3], [0, 2], [2, 4], [0, 13], [5, 13], [6, 13], [5, 7], [7, 9], [6, 8], [8, 10], [12, 13], [11, 13]]

class CLR_config:
    # video config
    IMAGE_SIZE = (200, 200)
    HEATMAP_SIZE = (50, 50)
    MAX_VIDEOLEN = 300
    NUM_JOINTS = 14
    SIGMA = 1
    CLIP_LENGTH = 16
    SLIDING_STRIDE= 8
    DATA_PATH = "../CLSR_DATA/public_dataset/"
    JOINTS_CHANNEL = 2

    # sentence config
    Time_Step = int((MAX_VIDEOLEN - CLIP_LENGTH) / SLIDING_STRIDE + 1)
    #sentence config
    ISPAD = True
    VOC_SZIE = 252  # 178+eos+sos+pad
    MAX_SENTENCELEN=11 # 5+EOS

    # net config
    VIDEO_FEATURE_DIM = 4096
    SEMANTIC_HIDDEN_DIM = 1024
    LEARNING_RATE = 0.0005
    Epochs = 100


class PHOENIX_cofig:
    # video config
    IMAGE_SIZE = (200, 200)
    NUM_JOINTS = 14
    HEATMAP_SIZE = (50, 50)
    MAX_VIDEOLEN = 200
    CLIP_LENGTH = 16
    SIGMA = 1
    SLIDING_STRIDE = 8
    VIDEO_FEATURE_DIM = 4096
    SEMANTIC_HIDDEN_DIM = 1024
    JOINTS_CHANNEL = 2

    # sentence config
    ISPAD = True
    MAX_SENTENCELEN = 29
    VOC_SZIE = 1234
    Time_Step = int(((MAX_VIDEOLEN - CLIP_LENGTH) / SLIDING_STRIDE) + 1)
    Epochs = 100
    LEARNING_RATE = 0.0001
    DATA_PATH = "../CLSR_DATA/phoenix2014-release/"


class PHOENIXT_cofig:
    # video config
    IMAGE_SIZE = (200, 200)
    NUM_JOINTS = 14
    HEATMAP_SIZE = (50, 50)
    MAX_VIDEOLEN = 200
    CLIP_LENGTH = 16
    SIGMA = 1
    SLIDING_STRIDE = 8
    VIDEO_FEATURE_DIM = 4096
    SEMANTIC_HIDDEN_DIM = 1024
    JOINTS_CHANNEL = 2

    # sentence config
    ISPAD = True
    MAX_SENTENCELEN = 53 #52+EOS
    VOC_SZIE = 2890 #2887+PAD+EOS+SOS
    Time_Step = int(((MAX_VIDEOLEN - CLIP_LENGTH) / SLIDING_STRIDE) + 1)
    Epochs = 100
    LEARNING_RATE = 0.001
    DATA_PATH = "../CLSR_DATA/PHOENIX-2014-T-release-v3/"
