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
from tensorflow.python import pywrap_tensorflow

#Preparing the Data
np.random.seed(36)

question_bank_path = os.path.join("data", "question_bank.tsv")
test_file_path = os.path.join("data", "test_set.tsv")
train_file_path = os.path.join("data", "training.tsv")
model_path = os.path.join("model", "model.pt")
result_path = os.path.join("result", "result.txt")
temp_path = os.path.join("result", "temp.txt")
answer_path = os.path.join("result", "answer.txt")
inputs_path = os.path.join("prepared_inputs.txt")

question_bank = pd.read_csv(question_bank_path, sep='\t', header=0)
test_file = pd.read_csv(test_file_path, sep='\t', header=None)
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
'''
new_neg_sentA = text_list[:, :1]
new_neg_sentB = text_list[:, 1:]
np.random.shuffle(new_neg_sentB)

for i in range(new_neg_sentB.shape[0]):
    if new_neg_sentB[i][0] == text_list[i][1]:
        b = text_list[(i + SHIFT_OFFSET) % new_neg_sentB.shape[0]][1]

neg_Sample = np.c_[new_neg_sentA, new_neg_sentB]

for b, l in zip(neg_Sample, text_list[:, 1:]):
    if b[0] == l[0]:
        print("false neg sample founded")
'''

NUM_TRAIN_SAMPLE = int(0.9 * text_list.shape[0])
NUM_DEV_SAMPLE = text_list.shape[0] - NUM_TRAIN_SAMPLE
NEG_SAMPLE_RITO = 0.5
NUM_NEG_SAMPLE = int(NEG_SAMPLE_RITO * NUM_TRAIN_SAMPLE)


# save
def save(model, optimizer):
    # save
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_path)


#????????????90???training.tsv??????????????????train set???????????????????????????query???????????????50???negative samples,
#negative samples???snetence2????????????????????????query????????????
#??????negative samples???train set????????????Labels????????????????????????????????????X???Y?????????
#????????????10???training.tsv??????????????????dev set, ?????????????????????
def getXY(text_list, neg_Sample, neg_Sample_count, query_count):
    n = text_list.shape[0]
    sum1 = 0
    sum2 = 0
    for i1, i2 in zip(query_count, neg_Sample_count):
        np.random.shuffle(neg_Sample[sum1:sum1 + i1, :])
        text_list = np.r_[text_list, neg_Sample[sum2:sum2 + i2, :]]
        sum1 = sum1 + i1
        sum2 = sum2 + i2

    trainY = np.r_[np.full((n, 1), 0), np.full((text_list.shape[0] - n, 1), 1)]
    #shuffle
    concate_X = np.c_[text_list, trainY]
    np.random.shuffle(concate_X)
    text_list = concate_X[:, :2]
    trainY = concate_X[:, 2:]
    return text_list, trainY


#train data and labels
full_list, fullY = getXY(text_list, neg_Sample, neg_Sample_Count, query_count)

train_list = full_list[:NUM_TRAIN_SAMPLE, :]
trainY = fullY[:NUM_TRAIN_SAMPLE, :]

dev_list = full_list[NUM_TRAIN_SAMPLE:, :]
devY = fullY[NUM_TRAIN_SAMPLE:, :]

#?????????(6???15???)
sentences_A = train_list[:, :1]
sentences_B = train_list[:, 1:]
sentences_A = sentences_A.flatten().tolist()
sentences_B = sentences_B.flatten().tolist()
train_label = trainY.flatten().tolist()

#inputs.keys() := dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
train_inputs = tokenizer(sentences_A,
                         sentences_B,
                         return_tensors='pt',
                         max_length=512,
                         truncation=True,
                         padding='max_length')
train_inputs['labels'] = torch.LongTensor(train_label).T


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


optim = AdamW(model.parameters(), lr=5e-6)


def startTrain(train_inputs, batch_size, num_epochs):
    BATCH_SIZE = batch_size
    torchDataSet = buildDataset(train_inputs)
    train_Loader = DataLoader(torchDataSet,
                              batch_size=num_epochs,
                              shuffle=False)

    num_epochs = num_epochs
    num_train_steps = num_epochs * len(train_Loader)
    device = torch.device("cuda")
    #Build Model
    model.to(device)
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

dev_inputs = tokenizer(sentences_A,
                       sentences_B,
                       return_tensors='pt',
                       max_length=512,
                       truncation=True,
                       padding='max_length')
dev_inputs['labels'] = torch.LongTensor([dev_label]).T


def startEval(dev_inputs, batch_size):
    torchDataSet = buildDataset(dev_inputs)
    dev_Loader = DataLoader(torchDataSet, batch_size=batch_size, shuffle=False)
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
            #print(outputs)

    final_score = metric.compute()
    #??????????????????
    print('final score: ', final_score)


#test?????????????????????????????????question bank??????????????????????????????feed???model
#def getTop50(test_list, model, question_bank_list):
unranked_list = []

hyper_param = []
hyper_param.append({'batch_size': 10, 'epochs': 2})
'''
hyper_param.append({'batch_size': 2, 'epochs': 3})
hyper_param.append({'batch_size': 2, 'epochs': 9})
hyper_param.append({'batch_size': 3, 'epochs': 8})
hyper_param.append({'batch_size': 6, 'epochs': 3})
hyper_param.append({'batch_size': 8, 'epochs': 3})
hyper_param.append({'batch_size': 4, 'epochs': 2})
hyper_param.append({'batch_size': 4, 'epochs': 3})
hyper_param.append({'batch_size': 4, 'epochs': 9})
'''
for hp in hyper_param:
    startTrain(train_inputs, hp['batch_size'], hp['epochs'])
    device = torch.device('cuda')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    startEval(dev_inputs, 1)


def predInputs(query, question_bank_list):
    qeuries = query.repeat(len(question_bank_list))
    qeuries = qeuries.flatten().tolist()
    qids = question_bank_list[:, :1].flatten().tolist()
    question_bank_list = question_bank_list[:, 1:].flatten().tolist()
    inputs = tokenizer(qeuries,
                       question_bank_list,
                       return_tensors='pt',
                       max_length=512,
                       truncation=True,
                       padding='max_length')
    torchDataSet = buildDataset(inputs)
    pred_Loader = DataLoader(torchDataSet, batch_size=1, shuffle=False)
    loop = tqdm(pred_Loader, leave=True)
    result_question_list = []
    for batch, s in zip(loop, qids):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        t = getProbs(input_ids, attention_mask, token_type_ids, query, s)
        result_question_list.append(t)
        loop.set_description(f'prediction for {query}')
        loop.set_postfix(question=s)
    return result_question_list


def getResult():
    result = []
    print("start prediction")
    for q in test_list[:]:
        l = []
        l.append(q[0])
        results = predInputs(q, question_bank_list)
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


def getProbs(input_ids, attention_mask, token_type_ids, query, qids):
    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
    logits = outputs.logits
    probs = softmax(logits, dim=1)
    prob = probs[0]
    t = (query, qids, prob[0].item())
    return t


#result = getResult()

#??????negative sample???????????????????????????????????????????????????????????????????????????????????????
#neg sample(sentA???sentB)????????????sentAs feed???bert1 ???????????????sentBs feed???bert2???
#???????????????bert????????????????????????bert?????????????????????????????????pooling?????????????????????cosine similarity
#??????????????????????????????????????????????????????neg sample