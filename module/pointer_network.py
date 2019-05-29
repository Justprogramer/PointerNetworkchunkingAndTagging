# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as f

from module.feature import WordFeature
from util.static_utils import *


class PointerNetwork(nn.Module):
    def __init__(self, encoder, decoder, weight_size, word_emb_size, max_sentence_length,
                 num_rnn_units, feature_names, feature_size_dict, feature_dim_dict, require_grad_dict,
                 pretrained_embed_dict, bi_flag=True):
        super(PointerNetwork, self).__init__()
        self.encoder_hidden_size = num_rnn_units * 2 if bi_flag else num_rnn_units
        self.weight_size = weight_size
        self.word_emb_size = word_emb_size
        self.decoder_hidden_size = num_rnn_units
        self.encoder = encoder
        self.decoder = decoder

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

        self.chunk_length_embedding = nn.Embedding(max_sentence_length, weight_size)
        init_embedding(self.chunk_length_embedding.weight)

        self.W1 = nn.Linear(self.encoder_hidden_size, weight_size, bias=False)
        init_linear(self.W1)
        self.W2 = nn.Linear(self.word_emb_size, weight_size, bias=False)
        init_linear(self.W2)
        self.W3 = nn.Linear(self.word_emb_size, weight_size, bias=False)
        init_linear(self.W3)
        self.W4 = nn.Linear(self.decoder_hidden_size, weight_size, bias=False)
        init_linear(self.W4)
        self.vt1 = nn.Linear(weight_size, 1, bias=False)
        init_linear(self.vt1)
        self.vt2 = nn.Linear(weight_size, 1, bias=False)
        init_linear(self.vt2)

        self.loss = nn.BCELoss()

    def forward(self, feed_tensor_dict, encoder_outputs, encoder_hn):
        # word level feature
        word_feed_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            word_feed_dict[feature_name] = feed_tensor_dict[feature_name]
        word_feature = self.word_feature_layer(**word_feed_dict)
        # input：(bs,L,D)
        batch_size = word_feature.size(0)
        output_list = list()
        for batch_index in range(batch_size):
            # single input in batch_size
            single_input = word_feature[batch_index]
            sentence_length = single_input.size(0)
            encoder_output = encoder_outputs[batch_size]
            # Decoding states initialization(lstmCell)
            decoder_input = to_var(torch.zeros(1, self.word_emb_size))
            decoder_hidden = to_var(torch.zeros([1, self.decoder_hidden_size]))
            decoder_cell_state = encoder_hn[-1]

            output = torch.zeros(sentence_length)
            last_bond, bond = 0, 0
            while bond < sentence_length:
                if bond == 0:
                    encoder_hn0 = to_var(torch.zeros(1, self.decoder_hidden_size))
                    decoder_hidden, decoder_cell_state, _ = self.decoder.pointer_forward(decoder_input, encoder_hn0,
                                                                                         decoder_hidden,
                                                                                         decoder_cell_state)
                else:
                    decoder_hidden, decoder_cell_state, _ = self.decoder.pointer_forward(single_input[last_bond:bond],
                                                                                         encoder_hn[last_bond:bond],
                                                                                         decoder_hidden,
                                                                                         decoder_cell_state)
                # segment
                blend1 = self.W1(encoder_output[bond:])  # ( L - start, W)
                blend2 = self.W2(single_input[bond:])  # (L-start,W)
                blend3 = self.W3(single_input[bond])  # (W)
                blend4 = self.w4(decoder_hidden)  # (W)
                blend_sum = f.tanh(blend1 + blend2 + blend3 + blend4)
                out1 = self.vt1(blend_sum).squeeze()  # (L-start)
                out2 = self.vt2(
                    self.chunk_length_embedding(to_var(torch.LongTensor([bond - last_bond])))).squeeze()  # 1-d
                out = (out1 + out2).contiguous()
                out = f.log_softmax(out, -1)
                max_value, max_index = torch.max(out)
                last_bond = bond
                bond = max_index + bond
                output[bond] = torch.Tensor([1])
            output_list.append(output)
        output_list = torch.stack(output_list, 1)
        return output_list

    def loss(self, logits, mask, segments):
        """
        Args:
            logits: size=(batch_size, seq_len)
            mask: size=(batch_size, seq_len)
            segments: size=(batch_size, seq_len)
        """
        select = logits.masked_select(mask)
        segments = segments.masked_select(mask)
        return self.loss(select, segments)

    def predict(self, ptr_network_output, actual_lens):
        batch_size = ptr_network_output.size(0)
        segment_list = []
        for i in range(batch_size):
            segment_list.append(ptr_network_output[i].cpu().data.numpy()[:actual_lens[i].data])
        return segment_list
