from operator import mod
from typing import List
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from numpy.lib.arraypad import _pad_simple
from transformers.tokenization_bert import BertTokenizer
from transformers import BartModel, modelcard
import numpy as np
import codecs
import os
import pandas as pd
import keras as ks
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers.utils.dummy_pt_objects import MODEL_MAPPING
from transformers_keras import Bert
import tensorflow as tf
from transformers import BertForNextSentencePrediction
from torch.nn.functional import softmax

#Preparing the Data

question_bank_path = os.path.join("data", "question_bank.tsv")
test_file_path = os.path.join("data", "test.tsv")
train_file_path = os.path.join("data", "training.tsv")

question_bank = pd.read_csv(question_bank_path,
                            sep='\t',
                            header=0,
                            index_col=0)
test_file = pd.read_csv(test_file_path, sep='\t', header=0)
text_file = pd.read_csv(train_file_path, sep='\t', header=0)

#calculate max count of questions, which have the same query
t1 = pd.read_csv(train_file_path, sep='\t', header=0,
                 index_col=1).values.flatten().tolist()
t2 = list(set(t1))
t2.sort(key=t1.index)
SHIFT_OFFSET = max(list(map(lambda t: t1.count(t), t2)))

question_bank_list = question_bank.values.flatten()
test_list = test_file.values
text_list = text_file.values
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)

question_bank_tokens = list(
    map(lambda t: tokenizer.tokenize(t)[:511] + ['[SEP]'], question_bank_list))


#['hello world hello word'] --> ['[CLS]' 'hello' 'world' '[SEP]' 'hello' 'word' '[SEP]']
def addCLS(text_list):
    tokens = []
    for p in text_list:
        s1 = ['[CLS]'] + tokenizer.tokenize(p[0])[:511] + ['[SEP]']
        s2 = tokenizer.tokenize(p[1])[:511] + ['[SEP]']
        tokens.append(s1 + s2)
    return tokens


#rotate second colum of text file(training.tsv)
t_text_list = np.transpose(text_list)
neg_Sample = np.concatenate((t_text_list[0].reshape((t_text_list.shape[1]), 1),
                             np.append(t_text_list[1, SHIFT_OFFSET:],
                                       t_text_list[1, :SHIFT_OFFSET]).reshape(
                                           (t_text_list.shape[1], 1))),
                            axis=1)

np.random.seed(41)
np.random.shuffle(neg_Sample)

#np.random.shuffle(text_tokens_ids)

NUM_TRAIN_SAMPLE = int(0.9 * text_list.shape[0])
NUM_DEV_SAMPLE = text_list.shape[0] - NUM_TRAIN_SAMPLE
NEG_SAMPLE_RITO = 0.5
NUM_NEG_SAMPLE = int(NEG_SAMPLE_RITO * NUM_TRAIN_SAMPLE)

#build train with neg_sample
'''
[['Tell me about Obama family tree.'
  'are you interested in seeing barack obamas family']
 ['Tell me about Obama family tree.'
  'would you like to know barack obamas geneology']
 ['Tell me about Obama family tree.'
  'would you like to know about obamas ancestors']
 ...
 ['Information about bobcat'
  'do you want to know about the wars fought in afghanistan']
 ["I'm interested in InuYasha" 'how about some organizing tips']
 ['how to build a fence?' 'are you referring to bellevue washington']]
'''
train_list = np.r_[text_list[:NUM_TRAIN_SAMPLE], neg_Sample[:NUM_NEG_SAMPLE]]
#train y
trainY = np.r_[np.full((NUM_TRAIN_SAMPLE, 1), True),
               np.full((NUM_NEG_SAMPLE, 1), False)]
#shuffle
concate_X = np.c_[train_list, trainY]
np.random.shuffle(concate_X)
train_list = concate_X[:, :2]
trainY = concate_X[:, 2:]

#encode sententces pair, return list of
'''
{'input_ids': tensor([[  101,  2425,  2033,  2055,  8112,  2155,  3392,  1012,   102,  2024,
          2017,  4699,  1999,  3773, 13857,  8112,  2015,  2155,   102, padding........]]), 
          'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,padding........]]), 
          'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,padding........]])}
'''
train_encoded = list(
    map(
        lambda t: tokenizer.encode_plus(t[0],
                                        text_pair=t[1],
                                        max_length=512,
                                        padding='max_length',
                                        return_tensors='pt'), train_list))

##build dev with neg_sample
dev_list = np.r_[text_list[NUM_TRAIN_SAMPLE:],
                 neg_Sample[NUM_NEG_SAMPLE:int(NUM_NEG_SAMPLE * 1.1)]]
#dev y
devY = np.r_[np.full((NUM_DEV_SAMPLE, 1), True),
             np.full((int(NUM_NEG_SAMPLE * 0.1), 1), False)]
#shuffle
concate_X = np.c_[dev_list, devY]
np.random.shuffle(concate_X)
dev_tokens_ids = concate_X[:, :2]
devY = concate_X[:, 2:]
#encode sententces pair,
dev_encoded = list(
    map(
        lambda t: tokenizer.encode_plus(t[0],
                                        text_pair=t[1],
                                        max_length=512,
                                        padding='max_length',
                                        return_tensors='pt'), dev_list))

#Build Model
model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')

# a model's output is a tuple, we only need the output tensor containing
# the relationships which is the first item in the tuple
seq_relationship_logits = model(**train_encoded[2])[0]

# index 0: sequence B is a continuation of sequence A
# index 1: sequence B is a random sequence
probs = softmax(seq_relationship_logits, dim=1)

print(trainY[2])
print(seq_relationship_logits)
print(probs)