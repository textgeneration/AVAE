# project configuration
BATCH_SIZE = 64
Z_DIMENSION = 256
C_DIM = 2
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = 1000000
ITERATRATIONS = 40000
LOG_INTERVAL = 30
SAVE_INTERVAL = 50
EMBEDDING = 200
GPU = False
LAMBDA_C = 0.1
LAMBDA_Z = 0.1

#PATH SETTING
IMDB_PATH = "./.data/imdb/imdb_review_data(cleaned_text)2014.txt"
AMAZON_PATH = "./.data/amazon_review_text.txt"
AMAZON_LABEL_PATH = "./.data/amazon/amazon_review_label.txt"
AMAZON_UNLABEL_PATH = "./.data/amazon/amazon_review_unlabel.txt"
SST_PATH = './.data/sst/'
WORDVEC_PATH = './.data/'
DIC_AMAZON = ["Baby.csv", "Beauty.csv", "Sports & Outdoors.csv", "Toys & Games.csv", "Video Games"]
MODEL_PATH = './models/'
