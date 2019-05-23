from torchtext import data, datasets
from torchtext.vocab import GloVe
from torchtext.data import TabularDataset
import json
import csv
from torch.utils.data import Dataset
import os
import spacy
import multiprocessing
import time
import concurrent.futures
import torch
from tqdm import tqdm
from torch.nn import init
from torchtext.vocab import Vectors
import CONFIG

class Text_Dataset(data.Dataset):
    name = 'text data'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path_pos, whther_labelled, text_field, label_field, test=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        cnt = 0
        print("preparing examples...")
        if(whther_labelled):
            f_label = open(CONFIG.AMAZON_LABEL_PATH)
            f_label_text = [each for each in f_label.readlines()]
            with open(os.path.join(path_pos), 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    line = line.strip().split()
                    if(len(line)):
                        label_text = f_label_text[cnt].split()
                        if(label_text[1] == '1'):
                            label_text[1] = 'negative'
                        if(label_text[1] == '3'):
                            label_text[1] = 'neural'
                        if(label_text[1] == '5'):
                            label_text[1] = 'positive'
                        examples.append(data.Example.fromlist([line, CONFIG.DIC_AMAZON[int(label_text[0])] + " ---- " + label_text[1]], fields))
                        cnt += 1
                        #if(cnt > 99):
                        #    break
        else:
            with open(os.path.join(path_pos), 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    line = line.strip().split()
                    if (len(line)):
                        examples.append(data.Example.fromlist([line, 0], fields))
                        cnt += 1


        super(Text_Dataset, self).__init__(examples, fields, **kwargs)

    # 用来快速创建多个 Dataset 的方法.
    @classmethod
    def splits(cls, root='./data',
               train='train.csv', test='test.csv', **kwargs):
        return super(Text_Dataset, cls).splits(
            root=root, train=train, test=test, **kwargs)

    def process_csv_line(self, sample, test):
        text = sample["comment_text"]
        text = text.replace('\n', ' ')
        label = None
        if not test:
            label = [v for v in
                     map(int, sample[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]])]
        return text, label

class Read_Dataset:
    def __init__(self, multi_process = 2, file_preprocess = 'other'):
        self.emb_dim = CONFIG.EMBEDDING
        self.batch_num = CONFIG.BATCH_SIZE
        self.multi_process = multi_process
        using_gpu = True
        #self.train_data = self.Read_txt("G:\\coding\\controlled-multiclass-text-generation-master\\.data\\imdb_train.txt", self.multi_process)
        #self.train_data[len(self.train_data):len(self.train_data)] = self.Read_other(".\\.data\\imdb\\aclImdb\\train\\neg", self.multi_process)
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, fix_length=16)
        self.LABEL = data.Field(sequential=False)
        if(CONFIG.C_DIM == 2):
            self.train_vae = Text_Dataset(CONFIG.IMDB_PATH, None, self.TEXT, self.LABEL)
            #get SST
            f = lambda ex: len(ex.text) <= 15 and ex.label != 'neutral' and len(ex.text) >= 3
            self.train_avae, val_sst, test_sst = datasets.SST.splits(
            self.TEXT, self.LABEL, root=CONFIG.SST_PATH, fine_grained=False, train_subtrees=False,
            filter_pred=f
            )
        else:
            self.train_avae = Text_Dataset(CONFIG.AMAZON_PATH, True, self.TEXT, self.LABEL)
            self.train_vae = Text_Dataset(CONFIG.AMAZON_UNLABEL_PATH, False, self.TEXT, self.LABEL)

        #get vector
        self.vectors = GloVe('6B', dim=self.emb_dim, cache=CONFIG.WORDVEC_PATH)

        self.TEXT.build_vocab(self.train_vae, self.train_avae, vectors=self.vectors)
        self.LABEL.build_vocab(self.train_avae)
        self.n_vocab = len(self.TEXT.vocab.itos)
        self.train_iter_vae = data.Iterator(dataset=self.train_vae, shuffle=True, batch_size=self.batch_num, train=True, repeat=False, device=None)
        self.train_iter_avae = data.Iterator(dataset=self.train_avae, shuffle=True, batch_size=self.batch_num, train=True, repeat=False, device=None)

    def write_data(self, path, data):
        with open(path,'w',encoding='utf-8') as f:
            for sent in data:
                for word in sent:
                    f.write(word + ' ')
                f.write('\n')

    def Read_json(self):
        return

    def Read_txt(self, path, multi_process = 1):
        ret = []
        with open(os.path.join(path), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                str = line.strip().split()
                if(len(str)):
                    ret.append(str)
        return ret

    def Read_other(self, path, multi_process = 1):
        ret = []
        file_num = 0
        file_path = []
        for lists in os.listdir(path):
            sub_path = os.path.join(path, lists)
            if os.path.isfile(sub_path):
                file_num += 1
                file_path.append(sub_path)
        print("Number of files are: " + str(file_num))
        start_time = time.time()


        with concurrent.futures.ProcessPoolExecutor(max_workers=multi_process) as executor:
            futures = {executor.submit(self.Read_txt, file_path, int(file_num / multi_process) * i, min(int(file_num / multi_process) * (i+1), file_num)): i for i in range(0,multi_process)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    ret.extend(future.result())
                except TypeError as e:
                    print(e)
        '''
        ret.extend(self.Read_txt(file_path,0,len(file_path)))
        '''

        print ("Process pool  fle execution in " + str(time.time() - start_time), "seconds")
        return ret

    def __len__(self):
        return

    def __getitem__(self, item):
        return

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]

