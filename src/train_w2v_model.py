# coding=utf-8
from collections import defaultdict

import pandas as pd
import param
import util
from gensim.models import Word2Vec

############################ 加载数据 ############################
df_all = pd.read_csv(param.data_path + '/output/corpus/all_data.csv', encoding='utf8', nrows=param.train_num)
df_all['penalty'] = df_all['penalty'] - 1

############################ w2v ############################
documents = df_all['content'].values
util.log('documents number %d' % len(documents))

texts = [[word for word in document.split(' ')] for document in documents]
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] >= 5] for text in texts]

util.log('Train Model...')
w2v = Word2Vec(texts, size=param.w2v_dim, window=5, iter=15, workers=12, seed=param.seed)
w2v.save(param.data_path + '/output/model/w2v_12w.model')
util.log('Save done!')
