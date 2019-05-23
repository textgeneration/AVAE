from torchtext import data, datasets
from torchtext.vocab import GloVe
from torchtext.data import TabularDataset
import json
import csv
from torch.utils.data import Dataset
import os
import spacy
from tqdm import tqdm
import multiprocessing
import time
import concurrent.futures
import random


class utils:
    def __init__(self,emb_dim = 200, batch_num = 32, multi_process = 2, file_preprocess = 'other'):
        self.emb_dim = emb_dim
        self.batch_num = batch_num
        self.multi_process = multi_process

        #self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
        #self.TEXT.build_vocab(self.train_data, vectors=GloVe('6B', dim=emb_dim))

    def write_data(self, path, data):
        with open(path,'w',encoding='utf-8') as f:
            for sent in data:
                for word in sent:
                    f.write(word + ' ')
                f.write('\n')

    def write_csv(self, filename, data):
        with open(filename, 'w', newline='') as outf:
            '''
            dict_writer = csv.DictWriter(outf, data[0].keys())
            dict_writer.writerow(dict(zip(data[0].keys(), data[0].keys())))
            for row in data:
                # print(row)
                dict_writer.writerow(row)
            outf.close()
            '''
            writer = csv.writer(outf)
            writer.writerows(data)

    def read_txt(self, path, st, end):
        preprocess_nlp = spacy.load('en_core_web_sm')
        ret = []
        cnt = 0
        for file in path[st:end]:
            cnt += 1
            f = open(file, 'r', encoding='utf-8')
            str_all = f.read()
            str_all = preprocess_nlp(str_all)
            for sent in str_all.sents:
                if(3 <= len(sent) and len(sent) < 16):
                    ret.append([str(word) for word in sent])
            f.close()
            if(cnt % 100 == 0):
                print(cnt)
                if(cnt > 600):
                    break
        preprocess_nlp = 1
        return ret

    def read_csv(self, path, colomn):
        with open(path,'r',encoding='utf-8') as f:
            reader = csv.reader(f)
            ret = [rows[colomn] for rows in reader]
        return ret

    def read_file_list(self, path, multi_process = 1):
        file_num = 0
        file_path = []
        for lists in os.listdir(path):
            sub_path = os.path.join(path, lists)
            if os.path.isfile(sub_path):
                file_num += 1
                file_path.append(sub_path)
        return file_path, file_num

    def paragraph_to_sentence(self, preprocess_nlp, paragraphs):
        #preprocess_nlp is language model: spacy.load('en_core_web_sm')
        #return sentence list
        text = []
        for each_paragraph in paragraphs:
            sentence = preprocess_nlp(each_paragraph)
            for each in sentence.sents:
                if (3 <= len(each) and len(each) < 16):
                    text.append([str(word) for word in each])
        return text


        return ret

    def read_json(self, path):
        #Reading json file from given path and return list
        f = open(path, encoding='utf-8')
        json_object = json.load(f)
        return json_object

    def __len__(self):
        return

    def __getitem__(self, item):
        return

class Read_imdb_text:
    def __init__(self):
        return

    def read_imdb_train(self,path):
        tools = utils()
        file_path = tools.Read_other(path)
        text = []
        preprocess_nlp = spacy.load('en_core_web_sm')
        for each_file in tqdm(file_path):
            f = open(each_file, 'r', encoding='utf-8')
            str_all = f.read()
            str_all = preprocess_nlp(str_all)
            for sent in str_all.sents:
                if (3 <= len(sent) and len(sent) < 16):
                    text.append([str(word) for word in sent])
            f.close()

        return text




if __name__ == '__main__':
    tools = utils()
