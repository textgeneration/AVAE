import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from avae.dataset import *
from avae.model import RNN_VAE

import argparse
import random
import time


mb_size = CONFIG.BATCH_SIZE
z_dim = CONFIG.Z_DIMENSION
c_dim = CONFIG.C_DIM
h_dim = z_dim + c_dim
lr = CONFIG.LEARNING_RATE
lr_decay_every = CONFIG.LEARNING_RATE_DECAY
n_iter = CONFIG.ITERATRATIONS
log_interval = CONFIG.LOG_INTERVAL
path_imdb = CONFIG.IMDB_PATH

dataset = Read_Dataset(path_imdb)

#torch.manual_seed(int(time.time()))

model = RNN_VAE(
    len(dataset.TEXT.vocab.vectors), h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.TEXT.vocab.vectors, freeze_embeddings=True,
    gpu=CONFIG.GPU
)

def load_pretrained_model(model, path):
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    dict_update = {}
    for k, v in pretrained_dict.items():
        if v.size() == model_dict[k].size():
            dict_update[k] = v
        else:
            print(k)
    model_dict.update(dict_update)
    model.load_state_dict(model_dict)
    model.eval()

load_pretrained_model(model, CONFIG.MODEL_PATH + 'avae_' + str(CONFIG.C_DIM))


for count in range(0,20):
    z = model.sample_z_prior(1)
    c = model.sample_c_prior(1)
    _, c_idx = torch.max(c, dim=1)
    sample_idxs = model.sample_sentence(z, c)
    print('\nSentiment: {}'.format(dataset.idx2label(int(c_idx) + 1)))
    print('{}\n'.format(dataset.idxs2sentence(sample_idxs)))



