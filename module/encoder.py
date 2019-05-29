# -*-coding:utf-8-*-
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, num_rnn_units, num_layers, rnn_unit_type, dropout_rate, use_cuda, bi_flag=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_rnn_units = num_rnn_units
        self.num_layers = num_layers
        self.bi_flag = bi_flag
        self.run_unit_type = rnn_unit_type
        self.use_cuda = use_cuda

        self.dropout_rate = dropout_rate
        # feature dropout
        self.dropout_feature = nn.Dropout(self.dropout_rate)

        # encoder type
        if self.rnn_unit_type == 'rnn':
            self.rnn = nn.RNN(self.input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)
        elif self.rnn_unit_type == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)
        elif self.rnn_unit_type == 'gru':
            self.rnn = nn.GRU(self.input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)

        # encoder dropout
        self.dropout_rnn = nn.Dropout(self.dropout_rate)

    def forward(self, feats):
        feats = self.dropout_feature(feats)
        if self.rnn_unit_type == 'rnn' or self.rnn_unit_type == 'gru':
            output, hn = self.rnn(feats)
            output = output.transpose(1, 0).contiguous()  # [bs, max_len, rnn_units]
            output = self.dropout_rnn(output.view(-1, output.size(-1)))
            return output, hn
        elif self.rnn_unit_type == 'lstm':
            output, (hn, cn) = self.rnn(feats)
            output = output.transpose(1, 0).contiguous()  # [bs, max_len, lstm_units]
            output = self.dropout_rnn(output.view(-1, output.size(-1)))
            return output, (hn, cn)
