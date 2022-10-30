# -*- coding: utf-8 -*-

import json
import random
from transformers import BertTokenizer
from utils import *

random.seed(100)
tokenizer = None

def get_one_sample_features(one, max_seq_length=512):

    context = one["acticle"]
    outline = one["outline"]
    outline = outline.lower().replace(" ", "")

    context = tokenizer.tokenize(context)
    # print("------")
    # print(context)
    # print()

    label = [0] * len(context)
    for i, word in enumerate(context):
        temp = word.replace(" ##", "")
        temp = temp.replace("##", "")
        if temp in outline:
            label[i] = 1
    # a = "".join(context[i].replace(" ##", "").replace("##", "") for i in range(len(label)) if label[i] == 1)

    # print("-----")
    # print(context)
    # print(a)
    # print(outline)
    # print("".join(context[i] for i in range(len(label)) if label[i] == 1))
    # print(label)
    label = [0] + label

    encoded_dict = tokenizer.encode_plus(
        context,
        max_length=max_seq_length,
        return_overflowing_tokens=True,
        padding="max_length",
        return_token_type_ids=True
    )
    input_ids = encoded_dict["input_ids"]
    attention_mask = encoded_dict["attention_mask"]
    # print(attention_mask)

    if len(label) < max_seq_length:
        label = label + [0] * (max_seq_length - len(label))
    else:
        label = label[:max_seq_length]

    return [input_ids, attention_mask, label, one]


def convert_to_features(data):
    features_list = []
    for pre in tqdm(data):
        feature = get_one_sample_features(pre)
        if feature != 0:
            features_list.append(feature)
    print('get {} samples'.format(len(features_list)))
    return features_list

def prepare_bert_data(filename, model_type='../model'):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_type)

    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    random.shuffle(data)
    data = convert_to_features(data)
    train_data = data[:-1000]
    dev_data = data[-1000:]
    print(len(train_data))
    print(len(dev_data))

    if not os.path.exists('data/valid.obj'):
        dump_file(dev_data, 'data/valid.obj')
    if not os.path.exists('data/train.obj'):
        dump_file(train_data, 'data/train.obj')

prepare_bert_data("./data/data2gen.json")
