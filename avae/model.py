import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
import CONFIG

class RNN_VAE(nn.Module):

    def __init__(self, n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3, max_sent_len=15, pretrained_embeddings=None, freeze_embeddings=False, gpu=False):
        super(RNN_VAE, self).__init__()

        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx
        self.MAX_SENT_LEN = max_sent_len

        self.n_vocab = n_vocab
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.p_word_dropout = p_word_dropout

        self.gpu = gpu

        """
        Word embeddings layer
        """
        if pretrained_embeddings is None:
            self.emb_dim = h_dim
            self.word_emb = nn.Embedding(n_vocab, h_dim, self.PAD_IDX)
        else:
            self.emb_dim = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_embeddings)

            if freeze_embeddings:
                self.word_emb.weight.requires_grad = False

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.GRU(self.emb_dim, h_dim)
        self.q_mu = nn.Linear(h_dim, z_dim)
        self.q_logvar = nn.Linear(h_dim, z_dim)

        """
        Decoder is GRU with `z` and `c` appended at its inputs
        """
        self.decoder = nn.GRU(self.emb_dim+z_dim+c_dim, z_dim+c_dim)
        self.decoder_drop = nn.Dropout(p=0.3)
        self.decoder_fc = nn.Linear(z_dim+c_dim, n_vocab)

        """`
        Discriminator is CNN as in Kim, 2014
        """
        self.conv3 = nn.Conv2d(1, 100, (3, self.emb_dim))
        self.conv4 = nn.Conv2d(1, 100, (4, self.emb_dim))
        self.conv5 = nn.Conv2d(1, 100, (5, self.emb_dim))

        self.disc_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(300, c_dim)
        )

        self.discriminator = nn.ModuleList([
            self.conv3, self.conv4, self.conv5, self.disc_fc
        ])

        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = chain(
            self.encoder.parameters(), self.q_mu.parameters(),
            self.q_logvar.parameters()
        )

        self.decoder_params = chain(
            self.decoder.parameters(), self.decoder_fc.parameters()
        )

        self.vae_params = chain(
            self.word_emb.parameters(), self.encoder_params, self.decoder_params
        )
        self.vae_params = filter(lambda p: p.requires_grad, self.vae_params)

        self.discriminator_params = filter(lambda p: p.requires_grad, self.discriminator.parameters())

        """
        Use GPU if set
        """
        if self.gpu:
            self.cuda()

    def forward_encoder(self, inputs):
        """
        Inputs is batch of sentences: seq_len x mbsize
        """
        inputs = self.word_emb(inputs)
        #print(inputs)
        return self.forward_encoder_embed(inputs)

    def forward_encoder_embed(self, inputs):
        """
        Inputs is embeddings of: seq_len x mbsize x emb_dim
        """
        _, h = self.encoder(inputs, None)

        # Forward to latent
        #print(h)
        h = h.view(-1, self.h_dim)
        #print(h)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar

    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = Variable(torch.randn(self.z_dim))
        eps = eps.cuda() if self.gpu else eps
        return mu + torch.exp(logvar/2) * eps

    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(mbsize, self.z_dim))
        z = z.cuda() if self.gpu else z
        #print(z)
        return z

    def sample_c_prior(self, mbsize):
        """
        Sample c ~ p(c) = Cat([0.5, 0.5])
        """
        c = torch.from_numpy(np.random.multinomial(1, [1 / CONFIG.C_DIM]* CONFIG.C_DIM, mbsize).astype('float32'))
        c = c.cuda() if self.gpu else c
        #print(c)
        return c

    def forward_decoder(self, inputs, z, c):
        """
        Inputs must be embeddings: seq_len x mbsize
        """
        #print(inputs)
        dec_inputs = self.word_dropout(inputs)
        #print(dec_inputs)

        # Forward
        seq_len = dec_inputs.size(0)

        # 1 x mbsize x (z_dim+c_dim)
        init_h = torch.cat([z.unsqueeze(0), c.unsqueeze(0).cuda()], dim=2)
        #print(init_h)
        inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
        #print(inputs_emb.shape)
        inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)
        #print(inputs_emb.shape)

        outputs, _ = self.decoder(inputs_emb, init_h)
        outputs = self.decoder_drop(outputs)
        seq_len, mbsize, _ = outputs.size()

        #print(outputs)

        outputs = outputs.view(seq_len*mbsize, -1)
        y = self.decoder_fc(outputs)
        #print(y)
        y = y.view(seq_len, mbsize, self.n_vocab)

        return y

    def forward_discriminator(self, inputs):
        """
        Inputs is batch of sentences: mbsize x seq_len
        """
        inputs = self.word_emb(inputs)
        return self.forward_discriminator_embed(inputs)

    def forward_discriminator_embed(self, inputs):
        """
        Inputs must be embeddings: mbsize x seq_len x emb_dim
        """
        inputs = inputs.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim

        x3 = F.relu(self.conv3(inputs)).squeeze()
        x4 = F.relu(self.conv4(inputs)).squeeze()
        x5 = F.relu(self.conv5(inputs)).squeeze()

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

        x = torch.cat([x3, x4, x5], dim=1)

        y = self.disc_fc(x)

        #print(y)

        return y

    def forward(self, sentence, use_c_prior=True):
        """
        Params:
        -------
        sentence: sequence of word indices.
        use_c_prior: whether to sample `c` from prior or from `discriminator`.

        Returns:
        --------
        recon_loss: reconstruction loss of VAE.
        kl_loss: KL-div loss of VAE.
        """
        self.train()

        mbsize = sentence.size(1)

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: '<start> I want to fly <eos>'
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>'
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.cuda() if self.gpu else pad_words

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        mu, logvar = self.forward_encoder(enc_inputs)
        z = self.sample_z(mu, logvar)

        if use_c_prior:
            c = self.sample_c_prior(mbsize)
        else:
            c = torch.zeros(mbsize, CONFIG.C_DIM).cuda()
            _, pos = torch.max(self.forward_discriminator(sentence.transpose(0, 1)), 1)
            for i in range(0,c.size(0)):
                c[i][pos[i]] = 1


        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z, c)

        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
        )
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))

        return recon_loss, kl_loss, z

    def generate_sentences(self, batch_size, class_0 = False):
        """
        Generate sentences and corresponding z of (batch_size x max_sent_len)
        """
        samples = []
        cs = []

        for _ in range(batch_size):
            z = self.sample_z_prior(1)
            c = self.sample_c_prior(1)
            if(class_0):
                c = torch.zeros(c.size())
            samples.append(self.sample_sentence(z, c, raw=True))
            cs.append(c.long())

        X_gen = torch.cat(samples, dim=0)
        c_gen = torch.cat(cs, dim=0)

        return X_gen, c_gen

    def sample_sentence(self, z, c, raw=False, temp=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        self.eval()

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'

        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        h = torch.cat([z, c], dim=2)

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = []

        if raw:
            outputs.append(self.START_IDX)

        for i in range(self.MAX_SENT_LEN):
            emb = self.word_emb(word).view(1, 1, -1)
            emb = torch.cat([emb, z, c], 2)

            output, h = self.decoder(emb, h)
            output = self.decoder_drop(output)
            y = self.decoder_fc(output).view(-1)
            y = F.softmax(y/temp, dim=0)

            idx = torch.multinomial(y,num_samples=1)

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            if not raw and idx == self.EOS_IDX:
                break

            outputs.append(idx)

        # Back to default state: train
        self.train()

        if raw:
            outputs = Variable(torch.LongTensor(outputs)).unsqueeze(0)
            return outputs.cuda() if self.gpu else outputs
        else:
            return outputs

    def generate_soft_embed(self, mbsize, temp=1, class_0 = False):
        """
        Generate soft embeddings of (mbsize x emb_dim) along with target z
        and c for each row (mbsize x {z_dim, c_dim})
        """
        samples = []
        targets_c = []
        targets_z = []

        for _ in range(mbsize):
            z = self.sample_z_prior(1)
            c = self.sample_c_prior(1)
            if(class_0):
                c = torch.zeros(c.size())
            samples.append(self.sample_soft_embed(z, c, temp=1, downDropoutBN=True))
            targets_z.append(z)
            targets_c.append(c)

        X_gen = torch.cat(samples, dim=0)
        targets_z = torch.cat(targets_z, dim=0)
        _, targets_c = torch.cat(targets_c, dim=0).max(dim=1)

        return X_gen, targets_z, targets_c

    def sample_soft_embed(self, z, c, temp=1,downDropoutBN=False):
        """
        Sample single soft embedded sentence from p(x|z,c) and temperature.
        Soft embeddings are calculated as weighted average of word_emb
        according to p(x|z,c).
        """

        self.decoder_drop.eval() if downDropoutBN else self.eval()
        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'
        emb = self.word_emb(word).view(1, 1, -1)
        emb = torch.cat([emb, z, c], 2)

        h = torch.cat([z, c], dim=2)

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = [self.word_emb(word).view(1, -1)]

        for i in range(self.MAX_SENT_LEN):
            output, h = self.decoder(emb, h)
            output = self.decoder_drop(output)
            o = self.decoder_fc(output).view(-1)

            # Sample softmax with temperature
            y = F.softmax(o / temp, dim=0)

            # Take expectation of embedding given output prob -> soft embedding
            # <y, w> = 1 x n_vocab * n_vocab x emb_dim
            emb = y.unsqueeze(0) @ self.word_emb.weight
            emb = emb.view(1, 1, -1)

            # Save resulting soft embedding
            outputs.append(emb.view(1, -1))

            # Append with z and c for the next input
            emb = torch.cat([emb, z, c], 2)

        # 1 x 16 x emb_dim
        outputs = torch.cat(outputs, dim=0).unsqueeze(0)

        # Back to default state: train
        self.train()

        return outputs.cuda() if self.gpu else outputs

    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        if isinstance(inputs, Variable):
            data = inputs.data.clone()
        else:
            data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                     .astype('uint8')
        )

        if self.gpu:
            mask = mask.cuda()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)
