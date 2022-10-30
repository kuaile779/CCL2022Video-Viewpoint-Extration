
from model import Bert4Sum
from prepare_data import *
import torch
import time, jieba
from transformers import logging as lg
lg.set_verbosity_error()

model_type = "../model"
model = Bert4Sum(model_type)
model.load_state_dict(torch.load('checkpoint1.th', map_location='cpu'))
model.cuda()

tokenizer = BertTokenizer.from_pretrained(model_type)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_new_text(text):

    while '[' in text and ']' in text:
        a = text.index('[')
        b = text.index(']')
        if a<b and text[a+1: b].isdigit():
            dt = text[a: b + 1]
            text = text.replace(dt, '')
        else:
            break
    return text

def get_one_outline(context, max_seq_length=512):

    model.eval()
    with torch.no_grad():

        context = tokenizer.tokenize(context)
        encoded_dict = tokenizer.encode_plus(
            context,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            padding="max_length",
            return_token_type_ids=True
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = encoded_dict["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        input_ids = encoded_dict["input_ids"]
        attention_mask = encoded_dict["attention_mask"]

        input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()
        attention_mask = torch.LongTensor(attention_mask).unsqueeze(0).cuda()

        predictions = model([input_ids, attention_mask, None])
        predictions = predictions.cpu()
        predictions = predictions.squeeze(0)
        predictions = to_list(predictions)

    res = ""
    for i in range(len(tokens)):
        if predictions[i] >= 0.5 and tokens[i] not in ["[CLS]", "[SEP]", "[UNK]"]:
            temp = tokens[i].replace(" ##", "")
            temp = temp.replace("##", "")
            res += temp
    return res

def refine_outline(outline):
    if len(outline) > 5:
        olist = jieba.lcut(outline)
        res = ""
        for i in range(len(olist) - 1):
            if olist[i] not in olist[i+1:]:
                res += olist[i]
        res += olist[-1]
        return res
    else:
        return outline

def get_all_outline(filename="./data/result.jsonl"):

    file = open(filename, encoding='utf-8')
    print("Start update!")
    time_start = time.time()  # 记录开始时间
    while 1:
        line = file.readline()
        if not line:
            break
        pre = eval(line)
        url = pre["url"]
        key_points = pre["key_points"]

        index = 0
        for kp in key_points:
            outline = kp["outline"]
            new_outline = get_one_outline(outline)
            if new_outline != "":
                kp["outline"] = new_outline
            else:
                kp["outline"] = get_new_text(outline)

            kp["outline"] = refine_outline(kp["outline"])

        res = {"url": url, "key_points": key_points}
        with open('./data/new_result.jsonl', "a+") as outFile:
            outFile.write(json.dumps(res, ensure_ascii=False) + '\n')
    time_end = time.time()  # 记录结束时间
    time_sum = (time_end - time_start) / 60  # 计算的时间差为程序的执行时间，单位为秒/s
    print("End update! Time consume: {}".format(time_sum))

get_all_outline()



