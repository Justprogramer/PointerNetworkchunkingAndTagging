# -*-coding:utf-8-*-
from module.decoder import Decoder
from module.encoder import Encoder
from module.pointer_network import PointerNetwork


def init_decoder(configs):
    use_cuda = configs['model_params']['use_cuda']
    decoder = Decoder()
    if use_cuda:
        return decoder.cuda()
    return decoder


def init_encoder(configs):
    # init rnn parameters
    path_pretrain_list = configs['data_params']['path_pretrain']
    rnn_unit_type = configs['model_params']['rnn_type']
    num_rnn_units = configs['model_params']['rnn_units']
    num_layers = configs['model_params']['rnn_layers']
    bi_flag = configs['model_params']['bi_flag']
    dropout_rate = configs['model_params']['dropout_rate']
    use_cuda = configs['model_params']['use_cuda']
    word_emb_size = configs["model_params"]["embed_sizes"] if path_pretrain_list is None else None

    encoder = Encoder(input_dim=word_emb_size, num_rnn_units=num_rnn_units, num_layers=num_layers,
                      rnn_unit_type=rnn_unit_type, dropout_rate=dropout_rate, use_cuda=use_cuda, bi_flag=bi_flag)
    if use_cuda:
        return encoder.cuda()
    return encoder


def init_model(configs):
    encoder = init_encoder(configs)
    decoder = init_decoder(configs)
    weight_size = configs['model_params']['weight_size']
    path_pretrain_list = configs['data_params']['path_pretrain']
    num_rnn_units = configs['model_params']['rnn_units']
    bi_flag = configs['model_params']['bi_flag']
    use_cuda = configs['model_params']['use_cuda']
    word_emb_size = configs["model_params"]["embed_sizes"] if path_pretrain_list is None else None
    # todo max_sentence_length
    model = PointerNetwork(encoder=encoder, decoder=decoder, weight_size=weight_size, word_emb_size=word_emb_size,
                           max_sentence_length=0, decoder_emb_size=word_emb_size, hidden_size=num_rnn_units,
                           bi_flag=bi_flag)
    if use_cuda:
        return model.cuda()
    return model
