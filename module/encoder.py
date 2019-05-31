# -*-coding:utf-8-*-
import torch.nn as nn

from module.feature import WordFeature
from util.static_utils import init_lstm_weight


class Encoder(nn.Module):
    def __init__(self, num_rnn_units, num_layers, rnn_unit_type, feature_names, feature_size_dict,
                 feature_dim_dict, require_grad_dict, pretrained_embed_dict, dropout_rate, use_cuda, bi_flag=True):
        super().__init__()
        self.num_rnn_units = num_rnn_units
        self.num_layers = num_layers
        self.bi_flag = bi_flag
        self.rnn_unit_type = rnn_unit_type
        self.use_cuda = use_cuda

        self.feature_names = feature_names
        self.feature_size_dict = feature_size_dict
        self.feature_dim_dict = feature_dim_dict
        self.require_grad_dict = require_grad_dict
        self.pretrained_embed_dict = pretrained_embed_dict

        # word level feature layer
        self.word_feature_layer = WordFeature(
            feature_names=self.feature_names, feature_size_dict=self.feature_size_dict,
            feature_dim_dict=self.feature_dim_dict, require_grad_dict=self.require_grad_dict,
            pretrained_embed_dict=self.pretrained_embed_dict)
        self.rnn_input_dim = 0
        for name in self.feature_names:
            self.rnn_input_dim += self.feature_dim_dict[name]

        self.dropout_rate = dropout_rate
        # feature dropout
        self.dropout_feature = nn.Dropout(self.dropout_rate)

        # encoder type
        if self.rnn_unit_type == 'rnn':
            self.rnn = nn.RNN(self.rnn_input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)
        elif self.rnn_unit_type == 'lstm':
            self.rnn = nn.LSTM(self.rnn_input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)
        elif self.rnn_unit_type == 'gru':
            self.rnn = nn.GRU(self.rnn_input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)
        init_lstm_weight(self.rnn, self.num_layers)
        # encoder dropout
        self.dropout_rnn = nn.Dropout(self.dropout_rate)

    def forward(self, **feed_dict):
        # word level feature
        word_feed_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            word_feed_dict[feature_name] = feed_dict[feature_name]
        word_feature = self.word_feature_layer(**word_feed_dict)

        feats = self.dropout_feature(word_feature)
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
