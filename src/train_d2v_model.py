# coding=utf-8
import codecs
import subprocess
from collections import namedtuple

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression

import param
import util

############################ 加载数据 ############################
df_all = pd.read_csv(param.data_path + '/output/corpus/all_data.csv', encoding='utf8', nrows=param.train_num).reset_index()
df_all['penalty'] = df_all['penalty'] - 1


############################ 定义函数、类及变量 ############################
def run_cmd(cmd):
    print(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for t, line in enumerate(iter(process.stdout.readline, b'')):
        line = line.decode('utf8').rstrip()
        print(line)
    process.communicate()
    return process.returncode


SentimentDocument = namedtuple('SentimentDocument', 'words tags')


class Doc_list(object):
    def __init__(self, f):
        self.f = f
    def __iter__(self):
        for i,line in enumerate(codecs.open(self.f,encoding='utf8')):
            words = line.strip().split(' ')
            tags = [int(words[0][2:])]
            words = words[1:]
            yield SentimentDocument(words,tags)


############################ 准备数据 ############################
doc_f = codecs.open(param.data_path + '/output/corpus/doc_for_d2v_12w.txt', 'w', encoding='utf8')
for i, contents in enumerate(df_all.iloc[:param.train_num]['content']):
    words = []
    for word in contents.split(' '):
        words.append(word)
    tags = [i]
    if i % 10000 == 0:
        util.log('iter = %d' % i)
    doc_f.write(u'_*{} {}\n'.format(i, ' '.join(words)))
doc_f.close()

############################ dbow d2v ############################
d2v = Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=3, window=30, sample=1e-5, workers=8, alpha=0.025, min_alpha=0.025)
doc_list = Doc_list(param.data_path + '/output/corpus/doc_for_d2v_12w.txt')
d2v.build_vocab(doc_list)

df_lb = df_all['penalty']

for i in range(5):
    util.log('pass: ' + str(i))
    #     run_cmd('shuf alldata-id.txt > alldata-id-shuf.txt')
    doc_list = Doc_list(param.data_path + '/output/corpus/doc_for_d2v_12w.txt')
    d2v.train(doc_list, total_examples=d2v.corpus_count, epochs=d2v.iter)
    X_d2v = np.array([d2v.docvecs[i] for i in range(param.train_num)])
    scores = cross_val_score(LogisticRegression(C=3), X_d2v, df_lb, cv=5)
    util.log('dbow: ' + str(scores) + ' ' + str(np.mean(scores)))
d2v.save(param.data_path + '/output/model/dbow_d2v_12w.model')
util.log('Save done!')

############################ dm d2v ############################
d2v = Doc2Vec(dm=1, size=300, negative=5, hs=0, min_count=3, window=30, sample=1e-5, workers=8, alpha=0.025, min_alpha=0.025)
doc_list = Doc_list(param.data_path + '/output/corpus/doc_for_d2v_12w.txt')
d2v.build_vocab(doc_list)

df_lb = df_all['penalty']

for i in range(10):
    util.log('pass: ' + str(i))
    #     run_cmd('shuf alldata-id.txt > alldata-id-shuf.txt')
    doc_list = Doc_list(param.data_path + '/output/corpus/doc_for_d2v_12w.txt')
    d2v.train(doc_list, total_examples=d2v.corpus_count, epochs=d2v.iter)
    X_d2v = np.array([d2v.docvecs[i] for i in range(param.train_num)])
    scores = cross_val_score(LogisticRegression(C=3), X_d2v, df_lb, cv=5)
    util.log('dm: ' + str(scores) + ' ' + str(np.mean(scores)))
d2v.save(param.data_path + '/output/model/dm_d2v_12w.model')
util.log('Save done!')
