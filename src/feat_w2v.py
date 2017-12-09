# coding=utf-8
from collections import defaultdict

import numpy as np
import pandas as pd
import param
import util
from gensim.models import Word2Vec

############################ 加载数据 & 模型 ############################
df_all = pd.read_csv(param.data_path + '/output/corpus/all_data.csv', encoding='utf8', nrows=param.train_num)
df_all['penalty'] = df_all['penalty'] - 1
documents = df_all['content'].values
texts = [[word for word in document.split(' ')] for document in documents]
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] >= 5] for text in texts]

model = Word2Vec.load(param.data_path + '/output/model/w2v_12w.model')

############################ w2v ############################
util.log('Start get w2v feat..')
w2v_feat = np.zeros((len(texts), param.w2v_dim))
w2v_feat_avg = np.zeros((len(texts), param.w2v_dim))
i = 0
for line in texts:
    num = 0
    for word in line:
        num += 1
        vec = model[word]
        w2v_feat[i, :] += vec
    w2v_feat_avg[i, :] = w2v_feat[i, :] / num
    i += 1
    if i % 1000 == 0:
        util.log(i)

pd.DataFrame(w2v_feat).to_csv(param.data_path + '/output/feature/w2v/w2v_12w.csv', encoding='utf8', index=None)
pd.DataFrame(w2v_feat_avg).to_csv(param.data_path + '/output/feature/w2v/w2v_avg_12w.csv', encoding='utf8', index=None)
util.log('Save w2v and w2v_avg feat done!')