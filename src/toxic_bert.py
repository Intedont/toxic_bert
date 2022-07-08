import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import numpy as np

class Dataset(torch.utils.data.Dataset):
    '''Класс датасета. Токенизирует входной датафрейм и преобразует лейблы в int64'''

    def __init__(self, df, label_name, data_name):
        
        tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    
        self.labels = [label for label in df[label_name]] 
        self.data = [tokenizer(str(text), 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df[data_name]]
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], np.array(self.labels[idx], dtype=np.int64 )


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(self.__class__, self).__init__()
        
        self.bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()
        
    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
