# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as f
import torch


class Decoder(nn.Module):
    def __init__(self, input_dim, target_size, num_rnn_units, rnn_unit_type, dropout_rate,
                 use_cuda, conv_filter_sizes, conv_filter_nums):
        super().__init__()
        self.input_dim = input_dim
        self.target_size = target_size
        self.num_rnn_units = num_rnn_units
        self.rnn_unit_types = rnn_unit_type
        self.dropout_rate = dropout_rate
        self.use_cuda = use_cuda
        self.filter_sizes = conv_filter_sizes
        self.filter_nums = conv_filter_nums
        # CNN list
        self.word_encoders_convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.filter_nums, kernel_size=(filter_size, self.input_dim))
            for filter_size in range(1, self.filter_sizes + 1)
        ])
        self.decoder_input_dim = self.num_rnn_units + self.input_dim + self.filter_sizes * self.filter_nums
        self.rnn_layer = nn.LSTMCell(self.decoder_input_dim, self.num_rnn_units, bias=False) \
            if self.rnn_unit_types == "lstm" else nn.GRUCell(self.decoder_input_dim, self.num_rnn_units, bias=False)
        self.fc = nn.Linear(self.num_rnn_units, target_size, bias=False)
        self.softmax = f.log_softmax

    def conv_and_pool(self, x, conv):
        x = f.relu(conv(x))
        x = f.max_pool2d(x, x.size(2))
        return x.squeeze()

    def forward(self, input, encoder_hidden, decoder_hidden, decoder_cell_state):
        c_h = torch.mean(encoder_hidden, dim=0, keepdim=True).squeeze()
        input = input.unsqueeze(0).unsqueeze(0)
        c_x = [self.conv_and_pool(input, conv) for conv in self.word_encoders_convs]
        c_x = torch.cat(c_x, 0)
        input = input.squeeze()
        c_w = torch.mean(input, dim=0, keepdim=True).squeeze()
        decoder_input = torch.cat([c_w, c_h, c_x], 0).unsqueeze(0)
        feats, cell_state = self.rnn_layer(decoder_input, (decoder_hidden, decoder_cell_state))
        return feats, cell_state, self.softmax(self.fc(feats), -1)


if __name__ == '__main__':
    decoder = Decoder(50, 3, 50, "lstm", 0.5, True, 3, 40).cuda()
    print(decoder)
    input = torch.randn(5, 50).cuda()
    encoder_hidden = torch.randn(5, 50).cuda()
    decoder_hidden = torch.randn(1, 50).cuda()
    decoder_cell_state = torch.randn(1, 50).cuda()
    result = decoder(input, encoder_hidden, decoder_hidden, decoder_cell_state)
    print(result)
