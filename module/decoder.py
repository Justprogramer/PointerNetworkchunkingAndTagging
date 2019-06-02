# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as f

from module.feature import WordFeature
from util.static_utils import init_cnn_weight, to_var


class Decoder(nn.Module):
    def __init__(self, encoder, conv_filter_sizes, conv_filter_nums):
        super().__init__()
        self.encoder = encoder
        self.input_dim = self.encoder.rnn_input_dim
        self.num_rnn_units = self.encoder.num_rnn_units * 2 if self.encoder.bi_flag else self.encoder.num_rnn_units
        self.rnn_unit_types = self.encoder.rnn_unit_type
        self.dropout_rate = self.encoder.dropout_rate
        self.use_cuda = self.encoder.use_cuda
        self.filter_sizes = conv_filter_sizes
        self.filter_nums = conv_filter_nums

        self.feature_names = self.encoder.feature_names
        self.feature_size_dict = self.encoder.feature_size_dict
        self.feature_dim_dict = self.encoder.feature_dim_dict
        self.require_grad_dict = self.encoder.require_grad_dict
        self.pretrained_embed_dict = self.encoder.pretrained_embed_dict

        self.target_size = self.feature_size_dict['label']

        self.loss_function = nn.CrossEntropyLoss()
        # word level feature layer
        self.word_feature_layer = WordFeature(
            feature_names=self.feature_names, feature_size_dict=self.feature_size_dict,
            feature_dim_dict=self.feature_dim_dict, require_grad_dict=self.require_grad_dict,
            pretrained_embed_dict=self.pretrained_embed_dict)
        # CNN list
        self.word_encoders_convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.filter_nums, kernel_size=(filter_size, self.input_dim))
            for filter_size in range(1, self.filter_sizes + 1)
        ]).cuda()
        for cnn_layer in self.word_encoders_convs:
            init_cnn_weight(cnn_layer)

        self.decoder_input_dim = self.num_rnn_units + self.input_dim + self.filter_sizes * self.filter_nums
        self.rnn_layer = nn.LSTMCell(self.decoder_input_dim, self.num_rnn_units, bias=False) \
            if self.rnn_unit_types == "lstm" else nn.GRUCell(self.decoder_input_dim, self.num_rnn_units, bias=False)
        self.fc = nn.Linear(self.num_rnn_units, self.target_size, bias=False)
        self.softmax = f.log_softmax

    def conv_and_pool(self, x, conv):
        # print(x.size())
        x = f.relu(conv(x.unsqueeze(0).unsqueeze(0)))
        x = f.max_pool2d(x, x.size(2))
        return x.squeeze()

    def pointer_forward(self, word_feature, encoder_hidden, decoder_hidden, decoder_cell_state):
        # todo 单个输入没考虑到，等待修复
        c_h = torch.mean(encoder_hidden, dim=0, keepdim=True).squeeze()
        phrase_length = word_feature.size(0)
        if phrase_length < self.filter_sizes:
            c_x = self.conv_and_pool(word_feature, self.word_encoders_convs[phrase_length - 1]).repeat(
                self.filter_sizes)
        else:
            c_x = [self.conv_and_pool(word_feature, conv) for conv in self.word_encoders_convs]
            c_x = torch.cat(c_x, 0)
        c_w = torch.mean(word_feature, dim=0, keepdim=True).squeeze()
        decoder_input = torch.cat([c_w, c_h, c_x], 0).unsqueeze(0)
        feats, cell_state = self.rnn_layer(decoder_input, (decoder_hidden, decoder_cell_state))
        return feats, cell_state, self.softmax(self.fc(feats), -1)

    def forward(self, feed_tensor_dict, segments, encoder_outputs):
        word_feed_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            word_feed_dict[feature_name] = feed_tensor_dict[feature_name]
        word_feature = self.word_feature_layer(**word_feed_dict)

        # Decoding states initialization(lstmCell)
        tag_list = []
        batch_size = word_feature.size(0)
        actual_lens = torch.sum(feed_tensor_dict[self.feature_names[0]] > 0, dim=1).int()
        for i in range(batch_size):
            count = 0
            encoder_output = encoder_outputs[i]
            decoder_hidden = torch.cat(
                (encoder_output[-1, :self.encoder.num_rnn_units], encoder_output[0, self.encoder.num_rnn_units:]),
                0).view(1, -1)
            cell_state = to_var(torch.zeros(1, self.num_rnn_units), self.encoder.use_cuda)
            if torch.is_tensor(segments):
                segment = segments[i][:actual_lens[i].item()].cpu().data.numpy()
            else:
                segment = segments[i][:actual_lens[i].item()]
            seg_list = self.split(segment)
            for (start, end) in seg_list:
                decoder_hidden, cell_state, output = self.pointer_forward(word_feature[i][start:end],
                                                                          encoder_output[start:end],
                                                                          decoder_hidden,
                                                                          cell_state)
                tag_list.append(output)
        tag_list = torch.stack(tag_list, 1).squeeze(0)
        return tag_list

    def split(self, segment):
        seg_list = []
        start = 0
        if len(segment) == 1:
            seg_list.append((0, 1))
        else:
            for i, se in enumerate(segment):
                if i == 0:
                    continue
                if se == 1:
                    seg_list.append((start, i))
                    start = i
                if i == len(segment) - 1:
                    seg_list.append((start, len(segment)))
        return seg_list

    def loss(self, logits, tags):
        return self.loss_function(logits, tags)

    def predict(self, lstm_outputs, actual_lens, segments):
        output_size = lstm_outputs.size(0)
        batch_size = len(segments)
        tags = []
        tags_list = []
        _, arg_max = torch.max(lstm_outputs, dim=1)
        for i in range(output_size):
            tags_list.append(arg_max[i].item())
        start = 0
        for j in range(batch_size):
            if torch.is_tensor(segments):
                segment = segments[j][:actual_lens[j].item()].cpu().data.numpy()
            else:
                segment = segments[j][:actual_lens[j].item()]
            segs = self.split(segment)
            tag_num = len(segs)
            temp_list = []
            for tag, seg in zip(tags_list[start:start + tag_num], segs):
                temp_list += [tag] * (seg[1] - seg[0])
            tags.append(temp_list)
            start += tag_num
        return tags
