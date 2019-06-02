# -*-coding:utf-8-*-
import codecs
import json
import os
import time
from collections import Counter

import pandas as pd

import util.common_util as my_util

# 证据名称列表
evidence_list = list()
# 笔录正文字典 文件名:内容
content_dict = dict()
# 笔录中举证质证文本 文件名：内容
train_evidence_paragraph_dict = dict()
test_evidence_paragraph_dict = dict()
# 笔录中存在的证据对应关系 文件名:[举证方 Evidence(E) ,证据名称 Trigger(T) ,证实内容 Content(C), 质证意见 Opinion(O),质证方 Anti-Evidence(A)]
tag_dic = dict()
# 标签训练数据
train_data = list()
test_data = dict()
# 完整文档存储路径
train_content_path = os.path.join('.', os.path.join('data', 'train_content.json'))
test_content_path = os.path.join('.', os.path.join('data', 'test_content.json'))
# 完整标签存储路径
tag_path = os.path.join('.', os.path.join('data', 'tag.json'))
# 标签中出现的所有证据名称统计路径
evidence_path = os.path.join('.', os.path.join('data', 'evidence.json'))
# 文档中只包含举证质证段落存储路径
train_evidence_paragraph_path = os.path.join('.', os.path.join('data', 'train_evidence_paragraph.json'))
test_evidence_paragraph_path = os.path.join('.', os.path.join('data', 'test_evidence_paragraph.json'))
# 处理成句子标签格式存储路径
train_path = os.path.join('.', os.path.join('data', 'train.json'))
test_path = os.path.join('.', os.path.join('data', 'test.json'))

# 句子标签全为"O"的数量
o_count_sentence = 0

# 质证句最小长度
min_opinion_len = 3
# 举证句最小长度
min_evidence_len = 6


def analyse_data():
    analyse_data_excel_content()

    length = len(content_dict.values())
    train_content_keys = sorted(content_dict)[:int(length * 0.9)]
    test_content_keys = sorted(content_dict)[int(length * 0.9):]
    train_content, test_content = {}, {}
    for key in train_content_keys:
        train_content[key] = content_dict[key]
    for key in test_content_keys:
        test_content[key] = content_dict[key]
    dump_data(train_content, train_content_path)
    dump_data(test_content, test_content_path)

    # analyse_dir_document()
    analyse_data_excel_tags()
    extract_evidence_paragraph(train_content, "train")
    extract_evidence_paragraph(test_content, "test")
    train_paragraph_dict = reduce_sentence_length(train_evidence_paragraph_dict)
    test_paragraph_dict = reduce_sentence_length(test_evidence_paragraph_dict)
    create_data(train_paragraph_dict, "train")
    create_data(test_paragraph_dict, "test")


def save():
    # 保存处理的数据

    dump_data(tag_dic, tag_path)
    dump_data(evidence_list, evidence_path)
    dump_data(train_evidence_paragraph_dict, train_evidence_paragraph_path)
    dump_data(test_evidence_paragraph_dict, test_evidence_paragraph_path)
    dump_data(train_data, train_path)
    dump_data(test_data, test_path)


# 从excel中加载数据
def analyse_data_excel_content(title=None, content=None):
    if title is None and content is None:
        rows = pd.read_excel("./raw_data/文书内容.xls", sheet_name=0, header=0)
        for title, content in rows.values:
            title = my_util.format_brackets(title.strip())
            # print(title)
            analyse_data_excel_content(title, content)
    else:
        old_paragraphs = [paragraph for paragraph in my_util.split_paragraph(content)
                          if paragraph is not None and len(paragraph.strip()) > 0]
        new_paragraphs = list()
        new_paragraph = ""
        # 合并发言人段落
        for paragraph in old_paragraphs:
            if my_util.check_paragraph(paragraph):
                if new_paragraph is not None and len(new_paragraph) > 0:
                    if '\u4e00' <= paragraph[-1] <= '\u9fff':
                        paragraph += "。"
                    new_paragraphs.append(new_paragraph)
                new_paragraph = paragraph
            else:
                if '\u4e00' <= paragraph[-1] <= '\u9fff':
                    paragraph += "。"
                new_paragraph = new_paragraph + paragraph
        content_dict[title] = [
            [my_util.clean_text(sentence) for sentence in paragraph.split("。")
             if sentence is not None and len(sentence.strip()) > 0]
            for paragraph in new_paragraphs]
    return content_dict[title]


def extract_single_sentence_from_paragraph(paragraph):
    sentences = []
    if my_util.is_nan(paragraph):
        return sentences
    for sentence in paragraph.split("。"):
        if sentence is not None and len(sentence.strip()) > 0:
            for sen in sentence.split("；"):
                if sen is not None and len(sen.strip()) > 0:
                    sentences.append(my_util.clean_text(sen))
    return sentences


def reduce_sentence_length(evidence_paragraph_dict):
    new_evidence_dict = {}
    for d in evidence_paragraph_dict:
        sentences = []
        for paragraphs in evidence_paragraph_dict[d]:
            for paragraph in paragraphs:
                sentences.extend(extract_single_sentence_from_paragraph(paragraph))
        new_evidence_dict[d] = sentences
    return new_evidence_dict


# 从doc和docx中加载文档，暂不使用
# def analyse_dir_document():
#     listdir = os.listdir(Args.raw_file_path)
#     if not os.path.exists(Args.temp_file_path):
#         os.makedirs(Args.temp_file_path)
#     for file_name in listdir:
#         title = my_util.format_brackets(file_name.split(".")[0])
#         if title not in content_dict:
#             path = os.path.join(Args.raw_file_path, file_name)
#             if "docx" in file_name or "DOCX" in file_name:
#                 pass
#             else:
#                 try:
#                     word = wc.Dispatch('Word.Application')
#                     doc = word.Documents.Open(path)
#                     path = os.path.join(Args.temp_file_path, "temp.docx")
#                     doc.SaveAs(path, "16")  # 转化后路径下的文件
#                     doc.Close()
#                     word.Quit()
#                 except:
#                     print("《%s》 发生错误" % title)
#                     continue
#             content = docx2txt.process(path)
#             print("补充文档《%s》 成功" % title)
#             analyse_data_excel_content(title, content)


# 举证方 Evidence(E) 证据名称 Trigger(T) 证实内容 Content(C) 质证意见 Opinion(O) 质证方 Anti-Evidence(A)
def analyse_data_excel_tags():
    rows = pd.read_excel("./raw_data/证据关系对应.xls", sheet_name=0, header=0)
    for title, E, T, C, V, A in rows.values:
        title = my_util.clean_text(title)
        E = my_util.clean_text(E)
        A = my_util.clean_text(A)
        title = my_util.format_brackets(title)
        # print("tag_title:%s" % title)
        T = extract_single_sentence_from_paragraph(T)
        C = extract_single_sentence_from_paragraph(C)
        V = extract_single_sentence_from_paragraph(V)
        if title not in tag_dic:
            tag_list = list()
            for t in T:
                tag_list.append([E, t, C, V, A])
            tag_dic[title] = tag_list
        else:
            for t in T:
                tag_dic[title].append([E, t, C, V, A])
                if t not in evidence_list:
                    evidence_list.append(t)


# 抽取主要举证质证段落
def extract_evidence_paragraph(content, type=None):
    for d in content:
        if d not in tag_dic:
            continue
        start, end = my_util.check_evidence_paragraph(content[d])
        print(
            "提取证据段落完成《%s》(%s)，起始位置：%s,结束位置：%s\n%s\n%s" % (
                d, len(content_dict[d]), start, end, content_dict[d][start],
                content_dict[d][end - 1]))
        if type == "train":
            train_evidence_paragraph_dict[d] = content[d][start:end]
        else:
            test_evidence_paragraph_dict[d] = content[d][start:end]


# 先调用load_data [word,segment,tag]
def create_data(evidence_paragraph_dict, type=None):
    global o_count_sentence
    if type == "train":
        data_list = train_data
    else:
        data_list = test_data

    for d in evidence_paragraph_dict:
        if d not in tag_dic:
            with codecs.open("log.txt", "a", "utf-8") as f:
                f.write("文档《%s》没有对应的数据标签\n" % d)
            continue
        evidence_content = evidence_paragraph_dict[d]
        for sentence in evidence_content:
            tag = ["O"] * len(sentence)
            segment = [1] + [0] * (len(sentence) - 1)
            has_t, has_c, has_v = False, False, False
            for [E, t, C, V, A] in tag_dic[d]:
                find_t = str(sentence).find(t)
                if find_t != -1 and tag[find_t] == "O":
                    has_t = True
                    segment[find_t] = 1
                    if find_t + len(t) < len(sentence):
                        segment[find_t + len(t)] = 1
                    tag = tag[:find_t] + ["T"] * len(t) + tag[find_t + len(t):]
                for c in C:
                    if len(c) <= 1:
                        continue
                    find_c = str(sentence).find(c)
                    if find_c != -1 and tag[find_c] == "O":
                        has_c = True
                        segment[find_c] = 1
                        if find_c + len(c) < len(sentence):
                            segment[find_c + len(c)] = 1
                        tag = tag[:find_c] + ["C"] * len(c) + tag[find_c + len(c):]
                for v in V:
                    if len(v) <= 1:
                        continue
                    find_v = str(sentence).find(v)
                    if find_v != -1 and tag[find_v] == "O":
                        has_v = True
                        segment[find_v] = 1
                        if find_v + len(v) < len(sentence):
                            segment[find_v + len(v)] = 1
                        tag = tag[:find_v] + ["V"] * len(v) + tag[find_v + len(v):]
                if len(tag) - Counter(tag)["O"] >= min_evidence_len:
                    find_e = str(sentence).find(E + "：")
                    if find_e != -1 and (has_t or has_c):
                        segment[find_e] = 1
                        if find_e + len(E) < len(sentence):
                            segment[find_e + len(E)] = 1
                        tag = tag[:find_e] + ["E"] * (len(E) + 1) + tag[find_e + len(E) + 1:]
                        continue
                if len(tag) - Counter(tag)["O"] >= min_opinion_len:
                    find_a = str(sentence).find(A + "：")
                    if find_a != -1 and has_v:
                        segment[find_a] = 1
                        if find_a + len(A) < len(sentence):
                            segment[find_a + len(A)] = 1
                        tag = tag[:find_a] + ["E"] * (len(A) + 1) + tag[find_a + len(A) + 1:]
            tag = reduce_tag(segment, tag)
            if type == "train":
                if not (has_v or has_t or has_c):
                    if o_count_sentence % 10 == 0:
                        data_list.append(([word for word in sentence], segment, tag))
                    o_count_sentence += 1
                else:
                    data_list.append(([word for word in sentence], segment, tag))
            else:
                # test 全部保存,按文档保存
                if d in data_list:
                    data_list[d].append(([word for word in sentence], segment, tag))
                else:
                    data_list[d] = list()
                    data_list[d].append(([word for word in sentence], segment, tag))


def reduce_tag(segment, tag):
    result = []
    for i, seg in enumerate(segment):
        if seg == 1:
            result.append(tag[i])
    return result


# 保存数据
def dump_data(data, path):
    with codecs.open(path, "w", "utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False))


# 写入日志
def dump_log():
    with codecs.open("log.txt", "a", "utf-8") as f:
        f.write("excel文本数据：%s条\n" % len(content_dict))
        f.write("证据关系文本数量：%s条\n" % len(tag_dic))
        f.write("处理获取train语料数量：%s条\n" % len(train_data))
        f.write("处理获取test语料数量：%s条\n" % len(test_data))
        f.write("\n")


# 加载数据
def load_data(data_path):
    content = ""
    with codecs.open(data_path, "r", "utf-8", errors='ignore') as f:
        for line in f:
            content = content + line.strip()
    return json.loads(content)


def pre_process_raw_data(options):
    with codecs.open("log.txt", "a", "utf-8") as fp:
        fp.write("开始处理数据：[%s]\n" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = time.time()
    if options:
        analyse_data()
        save()
        dump_log()
    with codecs.open("log.txt", "a", "utf-8") as fp:
        fp.write("处理数据结束：[%s], 费时：[%s]\n" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), time.time() - start))


if __name__ == '__main__':
    pre_process_raw_data(True)
