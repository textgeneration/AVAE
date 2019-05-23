import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import tqdm
from avae.dataset import *
from avae.model import RNN_VAE
import CONFIG
import argparse

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

model = RNN_VAE(
    len(dataset.TEXT.vocab.vectors), h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.TEXT.vocab.vectors, freeze_embeddings=True,
    gpu=CONFIG.GPU
)

model.cuda()
model.train()

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


def main():
    # Annealing for KL term
    kld_start_inc = 3000
    kld_weight = 0.01
    kld_max = 0.15
    kld_inc = (kld_max - kld_weight) / (n_iter - kld_start_inc)
    trainer = optim.Adam(model.vae_params, lr=lr)

    it = 1
    while(it <= n_iter):
        for batch in tqdm(dataset.train_iter_vae):
            if(it >= n_iter):
                break
            inputs = batch.text.cuda()

            recon_loss, kl_loss, z = model.forward(inputs)
            loss = recon_loss + kld_weight * kl_loss

            # Anneal kl_weight
            if it > kld_start_inc and kld_weight < kld_max:
                kld_weight += kld_inc


            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)
            trainer.step()
            trainer.zero_grad()

            if it % log_interval == 0:
                print('Iter-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; Grad_norm: {:.4f};'
                      .format(it, loss, recon_loss.data, kl_loss.data, grad_norm))
            if it % 500 == 0:
                save_model(it)
            # Anneal learning rate
            new_lr = lr * (0.5 ** (it // lr_decay_every))
            for param_group in trainer.param_groups:
                param_group['lr'] = new_lr
            it += 1


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def save_model(it):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(model.state_dict(), 'models/vae_' + str(CONFIG.C_DIM) + '_'+  str(it))


if __name__ == '__main__':
    main()

