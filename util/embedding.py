#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
用于从预训练词向量构建embedding表
"""


def load_tencent_embed(path_embed, word2id_dict):
    import codecs
    import numpy as np
    word_dim = 0
    word_embed_dict = {}
    with codecs.open(path_embed, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                word_dim = int(line.rstrip().split()[1])
                continue
            tokens = line.rstrip().split(' ')
            if tokens[0] in word2id_dict:
                word_embed_dict[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            if len(word_embed_dict) == len(word2id_dict):
                break
    return word_embed_dict, word_dim


def load_embed(path_embed):
    """
    读取预训练的embedding
    Args:
        path_embed: str
    Returns:
        word_embed_dict: dict, 健: word, 值: np.array, vector
        word_dim: int, 词向量的维度
    """
    import numpy as np
    lines_num, dim = 0, 0
    word_embed_dict = {}
    # iw = []
    # wi = {}
    with open(path_embed, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            word_embed_dict[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            # iw.append(tokens[0])
    # for i, w in enumerate(iw):
    #     wi[w] = i
    return word_embed_dict, dim


def build_word_embed(word2id_dict, path_embed, seed=137):
    """
    从预训练的文件中构建word embedding表
    Args:
        word2id_dict: dict, 健: word, 值: word id
        path_embed: str, 预训练的embedding文件
    Returns:
        word_embed_table: np.array, shape=[word_count, embed_dim]
        exact_match_count: int, 精确匹配的词数
        fuzzy_match_count: int, 精确匹配的词数
        unknown_count: int, 未匹配的词数
    """
    import numpy as np
    # word2vec_model, word_dim = load_tencent_embed(path_embed, word2id_dict)
    word2vec_model, word_dim = load_embed(path_embed)
    word_count = len(word2id_dict) + 1  # 0 is for padding value
    np.random.seed(seed)
    scope = np.sqrt(3. / word_dim)
    word_embed_table = np.random.uniform(
        -scope, scope, size=(word_count, word_dim)).astype('float32')
    exact_match_count, fuzzy_match_count, unknown_count = 0, 0, 0
    for word in word2id_dict:
        if word in word2vec_model:
            word_embed_table[word2id_dict[word]] = word2vec_model[word]
            exact_match_count += 1
        elif word.lower() in word2vec_model:
            word_embed_table[word2id_dict[word]] = word2vec_model[word.lower()]
            fuzzy_match_count += 1
        else:
            print("unknow word:%s" % word)
            unknown_count += 1
    total_count = exact_match_count + fuzzy_match_count + unknown_count
    return word_embed_table, exact_match_count, fuzzy_match_count, unknown_count, total_count


if __name__ == '__main__':
    import time
    from util.common_util import *

    token2id_dict = read_bin("../data/alphabet/token2id_dict.pkl")["word"]
    print("开始加载embedding：[%s]" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = time.time()
    vectors, dim = load_tencent_embed(os.path.join("..", Args.embedding_path), token2id_dict)
    print("处理数据结束：[%s], 费时：[%s]" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), time.time() - start))
    print("vector size:%i" % dim)
