# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.nn import CrossEntropyLoss, BCELoss


class Bert4Sum(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_type)
        self.n_hidden = self.encoder.config.hidden_size
        self.prediction = nn.Linear(self.n_hidden, 1, bias=False)

    def forward(self, inputs):
        [input_ids, attention_mask, label] = inputs
        hidden = self.encoder(input_ids, attention_mask=attention_mask)[0]  # b*l*h
        hidden = self.prediction(hidden).squeeze(-1)  # b*l
        if label is None:
            return torch.sigmoid(hidden)
        loss_fct = BCELoss(reduction="mean")
        loss = loss_fct(torch.sigmoid(hidden), label)
        return loss

