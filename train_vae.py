import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from AVae.dataset import *
from AVae.model import RNN_VAE

import argparse


parser = argparse.ArgumentParser(
    description='Attributed Controlled VAE'
)

parser.add_argument('--gpu', default=True, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')

args = parser.parse_args()
args.save = True

mb_size = 32
z_dim = 64
h_dim = 512
lr = 1e-4
lr_decay_every = 1000000
n_iter = 20000
log_interval = 1000
z_dim = h_dim
c_dim = 2
cate_num = 15

dataset = SST_Dataset()

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=False,
    gpu=args.gpu
)

model.load_state_dict(torch.load('models/default para/vae.bin'))
model.eval()

def main():
    # Annealing for KL term
    kld_start_inc = 3000
    kld_weight = 0.01
    kld_max = 0.15
    kld_inc = (kld_max - kld_weight) / (n_iter - kld_start_inc)

    trainer = optim.Adam(model.vae_params, lr=lr)
    trainer = optim.Adam(model.vae_params_no_clf, lr=lr)

    f_positive = open('sample_z_p', 'w')
    f_negative = open('sample_z_n', 'w')

    for it in range(n_iter):
        inputs, labels = dataset.next_batch(args.gpu)

        recon_loss, kl_loss, clf_loss, z = model.forward(inputs, labels)
        loss = recon_loss + kld_weight * kl_loss + 0.2 * clf_loss

        for each in range(0, z.size(0)):
            if(labels[each] == 0):
                f_positive.write(str(z[each].tolist()))
                f_positive.write('\n')
            else:
                f_negative.write(str(z[each].tolist()))
                f_negative.write('\n')


        # Anneal kl_weight
        if it > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc


        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)
        trainer.step()
        trainer.zero_grad()

        if it % log_interval == 0:
            z = model.sample_z_prior(1)
            c = model.sample_c_prior(1)

            sample_idxs = model.sample_sentence(z, c)
            sample_sent = dataset.idxs2sentence(sample_idxs)

            print('Iter-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; Grad_norm: {:.4f}; Clf: {:.4f}'
                  .format(it, loss.data[0], recon_loss.data[0], kl_loss.data[0], grad_norm, clf_loss.data[0]))

            print('Sample: "{}"'.format(sample_sent))
            print()


        # Anneal learning rate
        new_lr = lr * (0.5 ** (it // lr_decay_every))
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr


def save_model():
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/vae.bin')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if args.save:
            save_model()

        exit(0)

    if args.save:
        save_model()
