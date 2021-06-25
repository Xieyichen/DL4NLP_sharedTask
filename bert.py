from operator import mod, ne
from re import T
from typing import List
from click.termui import progressbar
from numpy.lib.arraypad import _pad_simple
from transformers import BartModel, modelcard
import numpy as np
import codecs
import os
import pandas as pd
import keras as ks
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers.utils.dummy_pt_objects import MODEL_MAPPING
import tensorflow as tf
from transformers import BertTokenizer, BertForNextSentencePrediction, BertForPreTraining
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AdamW
from torch.utils.data import DataLoader
from datasets import load_metric

#Preparing the Data

question_bank_path = os.path.join("data", "question_bank.tsv")
test_file_path = os.path.join("data", "test_set.tsv")
train_file_path = os.path.join("data", "training.tsv")
model_path = os.path.join("model", "model.pt")
result_path = os.path.join("result", "result.txt")
temp_path = os.path.join("result", "temp.txt")
answer_path = os.path.join("result", "answer.txt")
inputs_path = os.path.join("prepared_inputs.txt")

question_bank = pd.read_csv(question_bank_path, sep='\t', header=0)
test_file = pd.read_csv(test_file_path, sep='\t', header=0)
text_file = pd.read_csv(train_file_path, sep='\t', header=0)

#calculate max count of questions, which have the same query
t1 = pd.read_csv(train_file_path, sep='\t', header=0,
                 index_col=1).values.flatten().tolist()
t2 = list(set(t1))
t2.sort(key=t1.index)
query_count = list(map(lambda t: t1.count(t), t2))
neg_Sample_Count = list(map(lambda t: int(t * 0.5), query_count))

SHIFT_OFFSET = max(query_count)

question_bank_list = question_bank.values
test_list = test_file.values
text_list = text_file.values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

#rotate second colum of text file(training.tsv)
t_text_list = np.transpose(text_list)
neg_Sample = np.concatenate((t_text_list[0].reshape((t_text_list.shape[1]), 1),
                             np.append(t_text_list[1, SHIFT_OFFSET:],
                                       t_text_list[1, :SHIFT_OFFSET]).reshape(
                                           (t_text_list.shape[1], 1))),
                            axis=1)

NUM_TRAIN_SAMPLE = int(0.9 * text_list.shape[0])
NUM_DEV_SAMPLE = text_list.shape[0] - NUM_TRAIN_SAMPLE
NEG_SAMPLE_RITO = 0.5
NUM_NEG_SAMPLE = int(NEG_SAMPLE_RITO * NUM_TRAIN_SAMPLE)

np.random.seed(41)


# save
def save(model, optimizer):
    # save
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_path)


#取百分之90的training.tsv的句子对作为train set，然后依次加入每种query个数百分之50的negative samples,
#negative samples由snetence2的列旋转后，依照query进行打乱
#加入negative samples的train set与对应的Labels合并后进行打乱，再拆分成X和Y两部分
#取百分之10的training.tsv的句子对作为dev set, 并进行上述操作
def getTrainXY(text_list, neg_Sample, neg_Sample_count, query_count, num):
    train_list = text_list[:num, :]
    sum1 = 0
    sum2 = 0
    for i1, i2 in zip(query_count, neg_Sample_count):
        np.random.shuffle(neg_Sample[sum1:sum1 + query_count[i1], :])
        train_list = np.r_[train_list,
                           neg_Sample[sum2:sum2 + neg_Sample_count[i2], :]]
        sum1 = sum1 + i1
        sum2 = sum2 + i2

    trainY = np.r_[np.full((num, 1), 0),
                   np.full((train_list.shape[0] - num, 1), 1)]
    #shuffle
    concate_X = np.c_[train_list, trainY]
    np.random.shuffle(concate_X)
    train_list = concate_X[:, :2]
    trainY = concate_X[:, 2:]
    return train_list, trainY


#train_y = np.full((text_file.shape[0]), 0)


def getDevXY(text_list, neg_Sample, neg_Sample_count, query_count, num):
    train_list = text_list[num:, :]
    n = train_list.shape[0]
    sum1 = 0
    sum2 = 0
    for i1, i2 in zip(query_count, neg_Sample_count):
        np.random.shuffle(neg_Sample[sum1:sum1 + query_count[i1], :])
        train_list = np.r_[train_list,
                           neg_Sample[sum2:sum2 + neg_Sample_count[i2], :]]
        sum1 = sum1 + i1
        sum2 = sum2 + i2

    trainY = np.r_[np.full((n, 1), 0),
                   np.full((train_list.shape[0] - n, 1), 1)]
    #shuffle
    concate_X = np.c_[train_list, trainY]
    np.random.shuffle(concate_X)
    train_list = concate_X[:, :2]
    trainY = concate_X[:, 2:]
    return train_list, trainY


#train data and labels
train_list, trainY = getTrainXY(text_list, neg_Sample, neg_Sample_Count,
                                query_count, NUM_TRAIN_SAMPLE)
#dev data and labels
dev_neg_sample_count = list(
    map(lambda t: max(1, int(0.1 * t)), neg_Sample_Count))
dev_list, devY = getDevXY(text_list, neg_Sample, dev_neg_sample_count,
                          query_count, NUM_TRAIN_SAMPLE)

#新方法(6月15日)
'''
r = np.c_[train_list, trainY]
np.savetxt(inputs_path, r,header="query question label", delimiter="\t", fmt="%s") 
'''

sentences_A = train_list[:, :1]
sentences_B = train_list[:, 1:]
sentences_A = sentences_A.flatten().tolist()
sentences_B = sentences_B.flatten().tolist()
train_label = trainY.flatten().tolist()

#inputs.keys() := dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
inputs = tokenizer(sentences_A,
                   sentences_B,
                   return_tensors='pt',
                   max_length=512,
                   truncation=True,
                   padding='max_length')
inputs['labels'] = torch.LongTensor(train_label).T


#Dataloader
class buildDataset(torch.utils.data.Dataset):
    def __init__(self, encodes):
        self.encodes = encodes

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx])
            for key, val in self.encodes.items()
        }

    def __len__(self):
        return len(self.encodes.input_ids)


BATCH_SIZE = 15
torchDataSet = buildDataset(inputs)
train_Loader = DataLoader(torchDataSet, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda")

#Build Model
model.to(device)

optim = AdamW(model.parameters(), lr=3e-7)

num_epochs = 3
num_train_steps = num_epochs * len(train_Loader)
#progress_bar = tqdm(range(num_train_steps))


def startTrain():
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_Loader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
            loss = outputs.loss

            loss.backward()
            optim.step()
            optim.zero_grad()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    save(model, optim)
    print("train finished")


sentences_A = dev_list[:, :1]
sentences_B = dev_list[:, 1:]
sentences_A = sentences_A.flatten().tolist()
sentences_B = sentences_B.flatten().tolist()
dev_label = devY.flatten().tolist()

inputs = tokenizer(sentences_A,
                   sentences_B,
                   return_tensors='pt',
                   max_length=512,
                   truncation=True,
                   padding='max_length')
inputs['labels'] = torch.LongTensor([dev_label]).T

torchDataSet = buildDataset(inputs)
dev_Loader = DataLoader(torchDataSet, batch_size=BATCH_SIZE, shuffle=True)


def startEval():
    model.eval()
    progress_bar = tqdm(dev_Loader, leave=True)
    metric = load_metric("accuracy")
    for batch in progress_bar:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=preds,
                         references=batch['labels'].flatten())

    final_score = metric.compute()
    #打印最终分数
    print('final score: ', final_score)


#test中所有句子对的第一句和question bank中所有句子进行组合，feed进model
#def getTop50(test_list, model, question_bank_list):
unranked_list = []

startTrain()
device = torch.device('cuda')
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optim.load_state_dict(checkpoint['optimizer_state_dict'])
model.to(device)
startEval()
device = torch.device('cpu')
model.to(device)


def getTop50(query, question):
    strOfnum = question[0].split('Q')[1]
    numb = int(strOfnum)
    if numb % 4000 == 0:
        print('test sentence finished')
    inputs = tokenizer(query,
                       question[1],
                       return_tensors='pt',
                       max_length=512,
                       truncation=True,
                       padding='max_length')
    outputs = model(**inputs)
    logits = outputs.logits
    probs = softmax(logits, dim=1)
    prob = probs[0]
    t = (query, question[0], prob[0].item())
    return t


#best_one := list[('sentenceA', 'sentenceB', prob) x len(sentence_A)]
#
#ranked_list := list[list[ ('sentenceA', 'sentenceB', prob) x 50 ] x len(sentence_A)]

from multiprocessing.dummy import Pool as ThreadPool


def getResult():
    result = []
    for q in test_list:
        l = []
        l.append(q[0])
        zip_obj = zip(q.repeat(len(question_bank_list)), question_bank_list)
        pool = ThreadPool(2)
        results = [
            pool.apply_async(getTop50, args=(query, question))
            for query, question in zip_obj
        ]
        results = [p.get() for p in results]
        ranked_list = sorted(results, key=lambda x: x[2], reverse=True)
        ranked_list = list(map(lambda t: t[1], ranked_list))
        top_50 = ranked_list[:50]
        l = l + top_50
        strline = q[0] + '\t'
        fileObject = open(answer_path, 'a')
        for w in top_50:
            strline = strline + w + ','
        strline = strline[:-1]
        strline = strline + '\n'
        fileObject.write(strline)
        fileObject.close()
        result.append(l)
    return result


result = getResult()
'''
list1 = list(map(lambda t: t[1:], result))
qs1 = list(map(lambda t: t[:1], result))

list2 = list(map(lambda w: w.split(','), answer_file.values[:, 1:]))
qs2 = answer_file.values[:, :1]
bool_map = []
if len(result) == answer_file.value.shape[0]:
    for l1, l2, q1, q2 in zip(list1, list2, qs1, qs2):
        if q1 == q2[0]:
            b = 0
            for t1, t2 in zip(l1, l2):
                if t1 == t2:
                    b = b + 1
            bool_map.append(b)
        else:
            print('queries unmatch')
else:
    print('lines unmatch')

ps = list(map(lambda x: b / 50, bool_map))
print(ps)
'''