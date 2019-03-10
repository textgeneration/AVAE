import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import csv

from AVae.dataset import SST_Dataset,Test_Dataset
from AVae.model import RNN_VAE


import argparse
import random
import time


parser = argparse.ArgumentParser(
    description='AVAE'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--model', default='AVae', metavar='',
                    help='choose the model: {`vae`, `AVae`}, (default: `AVae`)')

args = parser.parse_args()


mb_size = 32
z_dim = 32
h_dim = 512
lr = 1e-3
lr_decay_every = 1000000
n_iter = 20000
log_interval = 1000
z_dim = h_dim
c_dim = 2

dataset = SST_Dataset()
#dataset = Test_Dataset()

torch.manual_seed(int(time.time()))

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
    gpu=args.gpu
)

if args.gpu:
    model.load_state_dict(torch.load('models/default para/{}.bin'.format("AVae")))
else:
    model.load_state_dict(torch.load('models/default para/{}.bin'.format("AVae"), map_location=lambda storage, loc: storage))

model.eval()

for i in range(0,1):
    # Samples latent and conditional codes randomly from prior
    z = model.sample_z_prior(1)
    c = model.sample_c_prior(1)

    # Generate positive sample given z
    c[0, 0], c[0, 1] = 1, 0

    _, c_idx = torch.max(c, dim=1)
    sample_idxs = model.sample_sentence(z, c, temp=2)

    print('\nSentiment: {}'.format(dataset.idx2label(int(c_idx))))
    print('Generated: {}'.format(dataset.idxs2sentence(sample_idxs)))

    # Generate negative sample from the same z
    z = model.sample_z_prior(1)
    c = model.sample_c_prior(1)
    c[0, 0], c[0, 1] = 1, 0

    _, c_idx = torch.max(c, dim=1)
    sample_idxs = model.sample_sentence(z, c, temp=2)

    print('\nSentiment: {}'.format(dataset.idx2label(int(c_idx))))
    print('Generated: {}'.format(dataset.idxs2sentence(sample_idxs)))

    print()

'''
# Interpolation
c = model.sample_c_prior(1)

z1 = model.sample_z_prior(1).view(1, 1, z_dim)
z1 = z1.cuda() if args.gpu else z1

z2 = model.sample_z_prior(1).view(1, 1, z_dim)
z2 = z2.cuda() if args.gpu else z2

# Interpolation coefficients
alphas = np.linspace(0, 1, 5)

print('Interpolation of z:')
print('-------------------')

for alpha in alphas:
    z = float(1-alpha)*z1 + float(alpha)*z2

    sample_idxs = model.sample_sentence(z, c, temp=0.1)
    sample_sent = dataset.idxs2sentence(sample_idxs)

    print("{}".format(sample_sent))

print()
'''

def write_file():
    data = open("sample","w",newline='')
    csv_writer = csv.writer(data,delimiter='\t')
    csv_writer.writerow(["text", "label"])
    for i in range(0, 1000):
        # Samples latent and conditional codes randomly from prior
        z = model.sample_z_prior(1)
        c = model.sample_c_prior(1)

        # Generate positive sample given z
        c[0, 0], c[0, 1] = 1, 0

        _, c_idx = torch.max(c, dim=1)
        sample_idxs = model.sample_sentence(z, c, temp=0.9)

        csv_writer.writerow([format(dataset.idxs2sentence(sample_idxs)),0])


    for i in range(0, 1000):
        # Samples latent and conditional codes randomly from prior
        z = model.sample_z_prior(1)
        c = model.sample_c_prior(1)

        # Generate positive sample given z
        c[0, 0], c[0, 1] = 0, 1

        _, c_idx = torch.max(c, dim=1)
        sample_idxs = model.sample_sentence(z, c, temp=0.9)

        csv_writer.writerow([format(dataset.idxs2sentence(sample_idxs)),1])

if __name__ == '__main__':
    write_file()