# -*- coding: utf-8 -*-

import argparse
import torch

from model import Bert4Sum
from prepare_data import prepare_bert_data
from utils import *
import torch.distributed as dist
from transformers import logging as lg
lg.set_verbosity_error()

torch.manual_seed(100)
np.random.seed(100)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=3.0e-5)
parser.add_argument("--max_grad_norm", type=float, default=1)
parser.add_argument("--model_type", type=str, default="../model")

args = parser.parse_args()
model_type = args.model_type

data = load_file('data/train.obj')
valid_data = load_file('data/valid.obj')
batch_size = args.batch_size
model = Bert4Sum(model_type).cuda()
optimizer = torch.optim.AdamW(model.parameters(),
                              weight_decay=0.01,
                              lr=args.lr)


def get_shuffle_data():
    np.random.shuffle(data)
    return data

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def iter_printer(total, epoch):
    return tqdm(range(0, total, batch_size), desc='epoch {}'.format(epoch))


def train(epoch):
    model.train()
    train_data = get_shuffle_data()
    total = len(train_data)
    step = -1
    for i in iter_printer(total, epoch):
        step += 1
        input_ids = [x[0] for x in train_data[i:i + batch_size]]
        attention_mask =  [x[1] for x in train_data[i:i + batch_size]]
        label = [x[2] for x in train_data[i:i + batch_size]]

        input_ids = torch.LongTensor(input_ids).cuda()
        attention_mask = torch.LongTensor(attention_mask).cuda()
        label = torch.FloatTensor(label).cuda()

        loss = model([input_ids, attention_mask, label])
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        global logging_loss
        logging_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # if args.logging_steps > 0 and global_step % args.logging_steps == 0:
            #     print(f"Current global step: {global_step}")
            #     print(f"average loss of batch: {logging_loss / args.logging_steps}")
            #     logging_loss = 0


def evaluation(epoch):
    model.eval()
    total = len(valid_data)
    right = 0
    with torch.no_grad():
        for i in iter_printer(total, epoch):
            input_ids = [x[0] for x in valid_data[i:i + batch_size]]
            attention_mask = [x[1] for x in valid_data[i:i + batch_size]]
            label = [x[2] for x in valid_data[i:i + batch_size]]

            input_ids = torch.LongTensor(input_ids).cuda()
            attention_mask1 = torch.LongTensor(attention_mask).cuda()

            predictions = model([input_ids, attention_mask1, None])
            predictions = predictions.cpu()
            predictions = to_list(predictions)

            for j in range(len(label)):
                num = attention_mask[j].count(1)
                tnum = 0
                for k in range(attention_mask[j].count(1)):
                    if predictions[j][k] >= 0.5:
                        temp = 1
                    else:
                        temp = 0
                    if label[j][k] == temp:
                        tnum += 1
                right += tnum / num

    acc = 100 * right / total
    print('epoch {} eval acc is {}'.format(epoch, acc))
    return acc

logging_loss = 0.0
global_step = 0
best_acc = 0.0

t_total = len(data) // batch_size // args.gradient_accumulation_steps * args.epoch
print(" Total optimization steps = ", t_total)
for epo in range(args.epoch):
    train(epo)
    accuracy = evaluation(epo)
    if accuracy > best_acc:
        best_acc = accuracy
        with open('checkpoint.th', 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)
