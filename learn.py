import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer
import torch

test_file_path = os.path.join("data", "test.tsv")
test_file = pd.read_csv(test_file_path, sep='\t', header=0)
test_list = test_file.values
t = range(0, 10 * 10, 10)
print(t)
'''
sentences_A = test_list[:, :1].flatten()
sentences_B = test_list[:, 1:].flatten()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(sentences_A[0],
                   sentences_B[0],
                   return_tensors='pt',
                   max_length=512,
                   truncation=True,
                   padding='max_length')
#inputs.keys() := dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
'''
