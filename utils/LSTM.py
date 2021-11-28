import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from config import *
from torchsummary import summary

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Attention(nn.Module):
    def __init__(self, hidden_dim, decoder_dim):
        super(Attention, self).__init__()
        self.atten = nn.Linear(hidden_dim+decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, hidden, encoder_output):
        # hidden = [num_layers * num_directions=1, batch, hidden_size)]
        # encoder_outputs = [ batch size, src len, enc hid dim ]
        src_len = encoder_output.shape[1]
        # repeat decoder hidden state src_len times
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        # hidden = [batch size, src len, dec hid dim]
        energy = torch.tanh(self.atten(torch.cat((hidden, encoder_output), dim=2)))
        # energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)
        # attention= [batch size, src len]
        return F.softmax(attention, dim=1)


class SeqDecode(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(SeqDecode, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer =n_layers

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx= input_size-3)
        self.rnn = nn.LSTM(hidden_size*3, hidden_size, n_layers, batch_first=True)
        self.attention = Attention(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size*4, input_size)
        self.dropout = nn.Dropout(0, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.weight_ih_l0.data)
                nn.init.orthogonal_(m.weight_ih_l0.data)
                m.bias_ih_l0.data.fill_(0)
                m.bias_hh_l0.data.fill_(0)

    def forward(self, x, h, c, encoder_output):
        self.rnn.flatten_parameters()
        # hidden = [num_layers=2 * num_directions=1, batch=2, hidden_size=4096]
        # encoder_output = [batch=2, step=24, hidden_size=4096]
        # x = [batch]
        x = x.unsqueeze(1)
        # x embedded =[batch, step=1, hidden_size=4096]
        embed = self.dropout(self.embedding(x))
        # att = [batch size, src len]===> [batch, 1 src_len]
        att = self.attention(h, encoder_output).unsqueeze(1)
        # att = [batch, 1 src_len] bmm [batch, sen_len, enc_hid_dim]==> [batch size, 1, enc hid dim]
        att = torch.bmm(att, encoder_output)
        x = torch.cat((embed, att), dim=2)
        x, (h, c) = self.rnn(x, (h, c))
        x = self.out(torch.cat((x, att, embed), dim=2)).squeeze(1)
        return x, h, c


class LSTM_seq(nn.Module):
    def __init__(self, max_seq=10, input_size=4096, hidden_size=4096, class_num=10):
        super(LSTM_seq, self).__init__()
        self.max_seq =max_seq
        self.class_num = class_num
        self.hidden_size = hidden_size

        self.Bilstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.Bilstm2 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.SeqDecode = SeqDecode(self.class_num, hidden_size, n_layers=1)

        self.memory_cell1 = nn.Linear(hidden_size, hidden_size)
        self.memory_cell2 = nn.Linear(hidden_size, hidden_size)
        self.memory_cell3 = nn.Linear(hidden_size * 2, hidden_size)
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.weight_ih_l0.data)
                nn.init.orthogonal_(m.weight_ih_l0.data)
                m.bias_ih_l0.data.fill_(0)
                m.bias_hh_l0.data.fill_(0)

    def forward(self, x, label):
        # x=[batch, time_step, feature]
        #print(torch.norm(x, dim=2).tolist())
        self.Bilstm1.flatten_parameters()
        self.Bilstm2.flatten_parameters()
        self.lstm.flatten_parameters()
        batch, _, _ = x.shape
        x, (h, c) = self.Bilstm1(x)
        h = torch.tanh(self.memory_cell1(h))
        c = torch.tanh(self.memory_cell1(c))

        x, (h, c) = self.Bilstm2(x,(h,c))
        h = torch.tanh(self.memory_cell2(h))
        c = torch.tanh(self.memory_cell2(c))

        x, (h, c) = self.lstm(x,(h, c))

        h = torch.tanh(self.memory_cell3(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))).unsqueeze(0)
        c = torch.tanh(self.memory_cell3(torch.cat((c[-2, :, :], c[-1, :, :]), dim=1))).unsqueeze(0)

        w = torch.full(size=(batch,), fill_value=(self.class_num-2), dtype=torch.long).to(device)
        outputs = torch.zeros(self.max_seq, batch, self.class_num).to(device)
        for t in range(self.max_seq):
            w, h, c = self.SeqDecode(w, h, c, x)
            outputs[t] = w
            teacher_force = random.random() < 0.5
            top1 = w.max(1)[1].detach()
            #w = top1
            w = (label[:, t] if teacher_force else top1)
        return outputs.permute(1, 0, 2)