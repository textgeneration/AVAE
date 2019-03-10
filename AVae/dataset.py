from torchtext import data, datasets
from torchtext.vocab import GloVe
import torch

class BatchWrapper(object):
    """对batch做个包装，方便调用，可选择性使用"""
    def __init__(self, dl,  x_var="text", y_var=["label"]):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_var

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)

            if self.y_vars is not None:
                temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
                label = torch.cat(temp, dim=1).long()
            else:
                raise ValueError('BatchWrapper: invalid label')
            text = x[0]
            length = x[1]
            yield (text, label, length)

    def __len__(self):
        return len(self.dl)

class SST_Dataset:

    def __init__(self, emb_dim=200, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        f = lambda ex: len(ex.text) <= 15 and ex.label != 'neutral'

        train, val, self.test = datasets.SST.splits(
            self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False,
            filter_pred=f
        )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, self.test), batch_size=mbsize, device=-1,
            shuffle=True, repeat=False
        )
        self.train_iter = iter(self.train_iter)
        self.val_iter = iter(self.val_iter)
        self.test_iter = iter(self.test_iter)

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_test_batch(self, gpu=False):
        batch = next(self.test_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]

class SST_Dataset_D:

    def __init__(self, emb_dim=200, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        f = lambda ex: ex.label != 'neutral'

        train, val, self.test = datasets.SST.splits(
            self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False,
            filter_pred=f
        )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, self.test), batch_size=mbsize, device=-1,
            shuffle=True, repeat=True
        )
        self.train_iter = iter(self.train_iter)
        self.val_iter = iter(self.val_iter)
        self.test_iter = iter(self.test_iter)

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_test_batch(self, gpu=False):
        batch = next(self.test_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]

class Test_Dataset:

    def __init__(self, emb_dim=200, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)
        fields = [
            ("text", self.TEXT), ("label", self.LABEL)]

        # Only take sentences with length <= 15
        #f = lambda ex: len(ex.text) <= 15 and ex.label != 'neutral'

        self.test = data.TabularDataset(
            path="sample", format='tsv', fields=fields,skip_header=True
        )

        self.TEXT.build_vocab(self.test, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(self.test)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.test_iter = data.BucketIterator(
            self.test, batch_size=mbsize, device=-1,
            shuffle=True, repeat=False
        )

        self.test_iter = iter(self.test_iter)
        #self.val_iter = iter(self.val_iter)

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_test_batch(self, gpu=False):
        batch = next(self.test_iter)
        if gpu:
            return batch.text.cuda(), batch.label.cuda()
        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]

class REVIEW_Dataset:

    def __init__(self, emb_dim=50, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)
        f = lambda ex: len(ex.reviews_text) <= 45 and ex.reviews_rating != '' and len(ex.reviews_text) > 3

        review_field = [("categories", None),
                        ("reviews_rating", self.LABEL),
                        ("reviews_text", self.TEXT),
                        ("reviews_title", None)]
        train = TabularDataset(
            path="./.data/all_train_reviews.csv", format='csv',
            skip_header=True, fields=review_field,
            filter_pred=f
        )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        val = train
        test = train

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=mbsize, device=-1,
            shuffle=True, repeat=True
        )
        self.train_iter = iter(self.train_iter)
        self.val_iter = iter(self.val_iter)

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        # print(batch.label.cuda())
        # print(batch.text.cuda())

        if gpu:
            return batch.reviews_text.cuda(), batch.reviews_rating.cuda()

        return batch.reviews_text, batch.reviews_rating

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.reviews_text.cuda(), batch.reviews_rating.cuda()

        return batch.reviews_text, batch.reviews_rating

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]
