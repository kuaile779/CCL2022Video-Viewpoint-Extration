# -*- coding: utf-8 -*-

import json, math
import random, os
import logging, torch
import numpy as np
import pickle, csv
import collections
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
from transformers import BasicTokenizer
from dataclasses import dataclass
from typing import List, Dict
from transformers import BertTokenizer
from torch.utils.data import Dataset
from transformers.tokenization_utils import _is_whitespace, _is_punctuation, _is_control

np.random.seed(100)
tokenizer = None

@dataclass
class cvExample:
    id: int
    title: str
    url: str
    captions: str
    doc_tokens: List
    key_points: List = None
    start_positions: List = None
    end_positions: List = None
    answer_texts: List = None

@dataclass
class cvFeature:
    id: int
    pid: int
    p_mask: List
    paragraph_len: int
    tokens: List
    token_to_orig_map: List
    input_ids: List
    attention_mask: List
    token_type_ids: List
    start_positions: List[int] = None
    end_positions: List[int] = None
    is_impossible: bool = None


def dump_file(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)

def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_doc_token(context_text):

    def _is_chinese_char(cp):
        if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def customize_tokenizer(text: str, do_lower_case=True) -> List[str]:
        temp_x = ""
        for char in text:
            if _is_chinese_char(ord(char)) or _is_punctuation(char) or _is_whitespace(char) or _is_control(char):
                temp_x += " " + char + " "
            else:
                temp_x += char
        if do_lower_case:
            temp_x = temp_x.lower()
        return temp_x.split()

    doc_tokens = []
    char_to_word_offset = []
    raw_doc_tokens = customize_tokenizer(context_text, True)
    k = 0
    temp_word = ""
    for char in context_text:
        if _is_whitespace(char):
            char_to_word_offset.append(k - 1)
            continue
        else:
            temp_word += char
            char_to_word_offset.append(k)
        if temp_word.lower() == raw_doc_tokens[k]:
            doc_tokens.append(temp_word)
            temp_word = ""
            k += 1
    assert k == len(raw_doc_tokens)

    return doc_tokens, char_to_word_offset


def get_new_span(context, content, raw_start):
    s = context.index(content, raw_start)
    e = s + len(content) - 1
    news = s
    newe = e
    for i in range(s, -1, -1):
        if context[i] == '[':
            news = i
            break
    for i in range(e, len(context)-1):
        if context[i+1] == '[':
            newe = i
            break
    return news, newe


def improve_answer_span1(doc_tokens, input_start, input_end, orig_answer_text):
    """
    Returns tokenized answer spans that better match the annotated answer.
    """
    orig_answer, _ = get_doc_token(orig_answer_text)
    tok_answer_text = " ".join(orig_answer)

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def read_examples(filename: str, is_training: bool) -> List[cvExample]:

    example_list = []
    id = 0
    file = open(filename, encoding='utf-8')
    save_num = 0
    save_data2gen = []
    while 1:
        line = file.readline()
        if not line:
            break
        pre = eval(line)
        title = pre['title']
        url = pre['url']
        captions = pre['captions']

        doc_tokens, char_to_word_offset = get_doc_token(captions)
        # print(doc_tokens)

        key_points = None
        start_positions = None
        end_positions = None
        answer_texts = []
        if is_training:
            key_points = pre['key_points']
            start_positions = []
            end_positions = []
            answer_texts = []
            # print("-----------")
            for kp in key_points:
                content = kp['reference']['content']
                raw_start =  captions.index("[" + str(kp["begin"]) + "]")
                s, e = get_new_span(captions, content, raw_start)
                # print(captions[s:e+1])

                raw_cap = captions[s:e+1]
                outline = kp['outline']
                save_data2gen.append({"id": save_num, "acticle": raw_cap, "outline": outline})
                save_num += 1

                start_position = char_to_word_offset[s]
                end_position = char_to_word_offset[e]
                start_position, end_position = improve_answer_span1(doc_tokens, start_position, end_position, captions[s:e+1])
                # print(''.join(doc_tokens[start_position:end_position+1]))
                start_positions.append(start_position)
                end_positions.append(end_position)
                answer_texts.append(captions[s:e+1])

        example_list.append(cvExample(
            id=id,
            title=title,
            url=url,
            captions=captions,
            key_points=key_points,
            doc_tokens=doc_tokens,
            start_positions=start_positions,
            end_positions=end_positions,
            answer_texts=answer_texts,
        ))

        id += 1
    file.close()
    #
    if is_training:
        print('get {} data2gen: '.format(len(save_data2gen)))  # 65868
        with open('data2gen.json', "w", encoding="utf-8") as writer:
            writer.write(json.dumps(save_data2gen, indent=2, ensure_ascii=False))

    print('get {} examples'.format(len(example_list)))
    return example_list

def _improve_answer_span(doc_tokens, input_start, input_end, orig_answer_text):
    """
    Returns tokenized answer spans that better match the annotated answer.
    """
    global tokenizer
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)



def convert_single_example_to_features(example: cvExample, is_training, max_seq_length=512,
                                       max_title_length=64, doc_stride=128):
    features = []
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training:
        start_positions = []
        end_positions = []
        for i in range(len(example.start_positions)):
            start_position = orig_to_tok_index[example.start_positions[i]]
            if example.end_positions[i] < len(example.doc_tokens) - 1:
                end_position = orig_to_tok_index[example.end_positions[i] + 1] - 1
            else:
                end_position = len(all_doc_tokens) - 1
            (start_position, end_position) = _improve_answer_span(
                all_doc_tokens, start_position, end_position, example.answer_texts[i]
            )
            # print("".join(all_doc_tokens[start_position: end_position+1]))
            start_positions.append(start_position)
            end_positions.append(end_position)
    else:
        start_positions = None
        end_positions = None

    title_ids = tokenizer.encode(example.title, add_special_tokens=False, max_length=max_title_length, truncation=True)

    sequence_pair_added_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    assert sequence_pair_added_tokens == 3

    span_doc_tokens = all_doc_tokens
    spans = []
    # print(max_seq_length - doc_stride - len(title_ids) - sequence_pair_added_tokens)
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(
            title_ids,
            span_doc_tokens,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            padding="max_length",
            stride=max_seq_length - doc_stride - len(title_ids) - sequence_pair_added_tokens,
            truncation="only_second",
            return_token_type_ids=True
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(title_ids) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = encoded_dict["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        # print("---------")
        for i in range(paragraph_len):
            index = len(title_ids) + i + 2
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]
        # print(tokens)
        # print(tokens[len(title_ids) + 2:])
        # print(example.doc_tokens[tok_to_orig_index[len(spans) * doc_stride]: tok_to_orig_index[len(spans) * doc_stride + paragraph_len]])

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(title_ids) + 2
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or len(encoded_dict["overflowing_tokens"]) == 0:
            break
        else:
            span_doc_tokens = encoded_dict["overflowing_tokens"]

    for span in spans:
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        p_mask = np.array(span["token_type_ids"])
        p_mask = np.minimum(p_mask, 1)
        p_mask = 1 - p_mask
        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        current_start_positions = None
        current_end_positions = None
        span_is_impossible = None
        if is_training:
            current_start_positions = [0 for i in range(max_seq_length)]
            current_end_positions = [0 for i in range(max_seq_length)]
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            doc_offset = len(title_ids) + 2
            # print("-------------")
            for i in range(len(start_positions)):
                start_position = start_positions[i]
                end_position = end_positions[i]
                if start_position >= doc_start and end_position <= doc_end:
                    span_is_impossible = False
                    current_start_positions[start_position - doc_start + doc_offset] = 1
                    current_end_positions[end_position - doc_start + doc_offset] = 1
                    # print(''.join(span["tokens"][start_position - doc_start + doc_offset: end_position - doc_start + doc_offset + 1]))

            if 1 not in current_start_positions:  # Current Feature does not contain answer span
                span_is_impossible = True
                current_start_positions[cls_index] = 1
                current_end_positions[cls_index] = 1
            assert span_is_impossible is not None

        features.append(
            cvFeature(
                input_ids=span["input_ids"],
                attention_mask=span["attention_mask"],
                token_type_ids=span["token_type_ids"],
                p_mask=p_mask.tolist(),
                id=0,
                pid=0,
                paragraph_len=span["paragraph_len"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_positions=current_start_positions,
                end_positions=current_end_positions,
                is_impossible=span_is_impossible
            )
        )

    return features

def convert_to_features(examples, is_training):

    unique_id = 0
    feature_list = []
    for example in tqdm(examples):
        feature = convert_single_example_to_features(example, is_training)

        for f in feature:
            f.id = example.id
            f.pid = unique_id
            unique_id += 1

        feature_list.extend(feature)

    # if is_training:
    #     print("raw length: ", len(feature_list))
    #     new_feature_list = []
    #     for f in feature_list:
    #         if not f.is_impossible:
    #             new_feature_list.append(f)
    #     feature_list = new_feature_list

    print('get {} features'.format(len(feature_list)))
    return feature_list

def convert_features_to_dataset(features: List[cvFeature], is_training: bool) -> Dataset:
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_example_indexes = torch.tensor([f.id for f in features], dtype=torch.long)
    all_feature_indexes = torch.tensor([f.pid for f in features], dtype=torch.long)
    if is_training:
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        all_start_labels = torch.tensor([f.start_positions for f in features], dtype=torch.float)
        all_end_labels = torch.tensor([f.end_positions for f in features], dtype=torch.float)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_start_labels,
            all_end_labels,
            all_p_mask,
            all_is_impossible,
            all_example_indexes,
            all_feature_indexes
        )

    else:
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_p_mask,
            all_example_indexes,
            all_feature_indexes
        )
    return dataset

def prepare_bert_data(model_type='./model'):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_type)
    if not os.path.exists('./data/train_features.obj'):
        train_examples = read_examples('./data/new-train.jsonl', True)
        train_features = convert_to_features(train_examples, True)
        dump_file(train_examples, './data/train_examples.obj')
        dump_file(train_features, './data/train_features.obj')  # 10585 / 41003

    if not os.path.exists('./data/valid_features.obj'):
        valid_examples = read_examples('./data/new-valid.jsonl', False)
        valid_features = convert_to_features(valid_examples, False)
        dump_file(valid_examples, './data/valid_examples.obj')
        dump_file(valid_features, './data/valid_features.obj')  # 158 / 289

    if not os.path.exists('./data/dev_features.obj'):
        dev_examples = read_examples('./data/dev.jsonl', False)
        dev_features = convert_to_features(dev_examples, False)
        dump_file(dev_examples, './data/dev_examples.obj')
        dump_file(dev_features, './data/dev_features.obj')  # 158 / 289

    if not os.path.exists('./data/test_features.obj'):
        test_examples = read_examples('./data/test.jsonl', False)
        test_features = convert_to_features(test_examples, False)
        dump_file(test_examples, './data/test_examples.obj')
        dump_file(test_features, './data/test_features.obj')  # 158 / 289

# count = 0
# prepare_bert_data()  #
# print(count)