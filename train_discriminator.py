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
from tqdm import tqdm

mb_size = CONFIG.BATCH_SIZE
z_dim = CONFIG.Z_DIMENSION
c_dim = CONFIG.C_DIM
h_dim = z_dim + c_dim
lr = CONFIG.LEARNING_RATE
lr_decay_every = CONFIG.LEARNING_RATE_DECAY
n_iter = 5000
log_interval = CONFIG.LOG_INTERVAL
kl_weight_max = 0.4

# Specific hyperparams
lambda_c = CONFIG.LAMBDA_C
lambda_z = CONFIG.LAMBDA_Z


if(CONFIG.C_DIM == 2):
    dataset = Read_Dataset(CONFIG.IMDB_PATH)


model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.TEXT.vocab.vectors, freeze_embeddings=True,
    gpu=CONFIG.GPU
)

# Load pretrained base VAE with c ~ p(c)
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

load_pretrained_model(model,CONFIG.VAE_PATH)


def kl_weight(it):
    """
    Credit to: https://github.com/kefirski/pytorch_RVAE/
    0 -> 1
    """
    return (math.tanh((it - 3500)/1000) + 1)/2


def temp(it):
    """
    Softmax temperature annealing
    1 -> 0
    """
    return 1-kl_weight(it) + 1e-5  # To avoid overflow


def main():
    trainer_D = optim.Adam(model.discriminator_params, lr=lr)
    trainer_G = optim.Adam(model.decoder_params, lr=lr)
    trainer_E = optim.Adam(model.encoder_params, lr=lr)

    for each in range(0,10):
        for batch in tqdm(dataset.train_iter_avae):
            inputs = batch.text.cuda()
            labels = batch.label.cuda() - 1
            y_disc_real = model.forward_discriminator(inputs.transpose(0, 1))
            loss_s = F.cross_entropy(y_disc_real, labels)
            loss_D = loss_s
            loss_D.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(model.discriminator_params, 5)
            trainer_D.step()
            trainer_D.zero_grad()

    it = 0
    while(it < n_iter):
        for batch in tqdm(dataset.train_iter_avae):
            inputs = batch.text.cuda()
            labels = batch.label.cuda() - 1

            """ Update discriminator, eq. 11 """
            batch_size = inputs.size(1)
            # get sentences and corresponding z
            x_gen, c_gen  = model.generate_sentences(batch_size)
            _, target_c = torch.max(c_gen, dim=1)

            y_disc_real = model.forward_discriminator(inputs.transpose(0, 1))
            y_disc_fake = model.forward_discriminator(x_gen)

            log_y_disc_fake = F.log_softmax(y_disc_fake, dim=1)

            loss_s = F.cross_entropy(y_disc_real, labels)
            loss_D = loss_s

            loss_D.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(model.discriminator_params, 5)
            trainer_D.step()
            trainer_D.zero_grad()

            """ Update generator, eq. 8 """
            # Forward VAE with c ~ q(c|x) instead of from prior
            recon_loss, kl_loss, z = model.forward(inputs, use_c_prior=False)
            # x_gen: mbsize x seq_len x emb_dim
            x_gen_attr, target_z, target_c = model.generate_soft_embed(batch_size, temp=temp(it))

            # y_z: mbsize x z_dim
            y_z, _ = model.forward_encoder_embed(x_gen_attr.transpose(0, 1))
            y_c = model.forward_discriminator_embed(x_gen_attr)

            loss_vae = recon_loss + kl_weight_max * kl_loss
            loss_attr_c = F.cross_entropy(y_c, target_c)
            loss_attr_z = F.mse_loss(y_z, target_z)

            loss_G = loss_vae + lambda_c * loss_attr_c + lambda_z*loss_attr_z
            model.train()
            loss_G.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(model.decoder_params, 5)
            trainer_G.step()
            trainer_G.zero_grad()


            """ Update encoder, eq. 4 """
            recon_loss, kl_loss, _ = model.forward(inputs, use_c_prior=False)

            loss_E = recon_loss + kl_weight_max * kl_loss

            loss_E.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(model.encoder_params, 5)
            trainer_E.step()
            trainer_E.zero_grad()

            if it % log_interval == 0:
                z = model.sample_z_prior(1)
                c = model.sample_c_prior(1)

                sample_idxs = model.sample_sentence(z, c)
                sample_sent = dataset.idxs2sentence(sample_idxs)

                print('Iter-{}; loss_D: {:.4f}; loss_G: {:.4f}; reacon_loss: {:.4f}; kl_loss: {:.4f}; attr_c: {:.4f}; attr_z:{:.4f}'
                      .format(it, float(loss_D), float(loss_G), float(recon_loss), float(kl_weight_max * kl_loss), float(loss_attr_c), float(loss_attr_z)))

                _, c_idx = torch.max(c, dim=1)

                print('c = {}'.format(dataset.idx2label(c_idx + 1)))
                print('Sample: "{}"'.format(sample_sent))
                print('sample2: "{}"'.format(dataset.idxs2sentence(model.sample_sentence(z, c))))
            if it % 100 == 0:
                save_model(it)

            it += 1


def save_model(it):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/avae_'+ str(CONFIG.C_DIM) + '_'+ str(it))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if args.save:
            save_model()

        exit(0)

    if args.save:
        save_model()