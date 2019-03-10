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
    description='AVAE'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')

args = parser.parse_args()


mb_size = 32
lr = 1e-4
lr_decay_every = 1000000
n_iter = 10000
log_interval = 50
class_num = 15


# dataset = WikiText_Dataset()
# dataset = IMDB_Dataset()



class Clf(nn.Module):

    def __init__(self):
        super(Clf, self).__init__()

        emb_dim = dataset.get_vocab_vectors().size(1)
        self.word_emb = nn.Embedding(dataset.n_vocab, emb_dim)
        # Set pretrained embeddings
        self.word_emb.weight.data.copy_(dataset.get_vocab_vectors())
        self.word_emb.weight.requires_grad = False

        self.conv3 = nn.Conv2d(1, 100, (3, emb_dim))
        self.conv4 = nn.Conv2d(1, 100, (4, emb_dim))
        self.conv5 = nn.Conv2d(1, 100, (5, emb_dim))

        self.discriminator = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(300, class_num)
        )

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward(self, inputs):
        inputs = self.word_emb(inputs)
        inputs = inputs.unsqueeze(1)

        x3 = F.relu(self.conv3(inputs)).squeeze()
        x4 = F.relu(self.conv4(inputs)).squeeze()
        x5 = F.relu(self.conv5(inputs)).squeeze()

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

        x = torch.cat([x3, x4, x5], dim=1)

        y = self.discriminator(x)

        return y

args.save = True
dataset = SST_Dataset()
#dataset = Test_Dataset()
model = Clf()
#


def main():

    trainer = optim.Adam(model.trainable_parameters(), lr=lr, weight_decay=1e-4)

    args.gpu = True
    if args.gpu:
        model.cuda()

    model.train()
    max_acc = 0
    for it in range(n_iter):
        inputs, labels = dataset.next_batch(args.gpu)

        inputs = inputs.transpose(0, 1)  # mbsize x seq_len
        y = model.forward(inputs)

        loss = F.cross_entropy(y, labels)

        loss.backward()
        trainer.step()
        trainer.zero_grad()

        if it % log_interval == 0:
            accs = []

            # Test on validation
            for _ in range(30):
                inputs, labels = dataset.next_validation_batch(args.gpu)
                inputs = inputs.transpose(0, 1)

                _, y = model.forward(inputs).max(dim=1)

                acc = float((y == labels).sum()) / y.size(0)
                accs.append(acc)

            print('Iter-{}; loss: {:.4f}; val_acc: {:.4f}'.format(it, float(loss), np.mean(accs)))

            if np.mean(accs) > max_acc:
                max_acc = np.mean(accs)
                save_model()

def test_acc():
    model.load_state_dict(torch.load('models/clf.bin'))
    model.eval()

    accs = []
    for _ in range(math.ceil(len(dataset.test.examples)/mb_size)):
        inputs, labels = dataset.next_test_batch(args.gpu)
        inputs = inputs.transpose(0, 1)

        _, y = model.forward(inputs).max(dim=1)

        acc = float((y == labels).sum()) / y.size(0)
        accs.append(acc)
    print(torch.mean(torch.tensor(accs)))

def save_model():
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(model.state_dict(), 'models/clf.bin')

if __name__ == '__main__':
    test_acc()
    #main()