# coding=utf-8
import codecs

import jieba
import jieba.analyse
import jieba.posseg
import pandas as pd

import param
import util


############################ 定义分词函数 ############################
def split_word(text, stopwords):
    word_list = jieba.cut(text)
    start = True
    result = ''
    for word in word_list:
        word = word.strip()
        if word not in stopwords:
            if start:
                result = word
                start = False
            else:
                result += ' ' + word
    return result.encode('utf-8')


############################ 加载停用词 ############################
stopwords = {}
for line in codecs.open(param.data_path + '/input/stop.txt', 'r', 'utf-8'):
    stopwords[line.rstrip()] = 1

############################ 加载数据 & 分词 ############################
df_tr = []
for i, line in enumerate(open(param.data_path + '/input/train.txt')):
    if i % 1000 == 1:
        util.log('iter = %d' % i)
    segs = line.split('\t')
    row = {}
    row['id'] = segs[0]
    row['content'] = split_word(segs[1].strip(), stopwords)
    row['penalty'] = segs[2]
    row['laws'] = segs[3].strip()
    df_tr.append(row)
df_tr = pd.DataFrame(df_tr)

df_te = []
for i, line in enumerate(open(param.data_path + '/input/test.txt')):
    if i % 1000 == 1:
        util.log('iter = %d' % i)
    segs = line.split('\t')
    row = {}
    row['id'] = segs[0]
    row['content'] = split_word(segs[1].strip(), stopwords)
    df_te.append(row)
df_te = pd.DataFrame(df_te)

print(df_tr.shape)
print(df_te.shape)

############################ 写出数据 ############################
df_all = pd.concat([df_tr, df_te]).fillna(0)
df_all.to_csv(param.data_path + '/output/corpus/all_data.csv', index=None)
