# coding=utf-8
import re

import numpy as np
import pandas as pd

import param
import util

df_tr = []
util.log('For train.txt:')
for i, line in enumerate(open(param.data_path + '/input/train.txt')):
    if i % 1000 == 1:
        util.log('iter = %d' % i)
    segs = line.split('\t')
    row = {}
    row['id'] = segs[0]
    row['raw_content'] = segs[1].strip()
    df_tr.append(row)
df_tr = pd.DataFrame(df_tr)

df_te = []
util.log('For test.txt:')
for i, line in enumerate(open(param.data_path + '/input/test.txt')):
    if i % 1000 == 1:
        util.log('iter = %d' % i)
    segs = line.split('\t')
    row = {}
    row['id'] = segs[0]
    row['raw_content'] = segs[1].strip()
    df_te.append(row)
df_te = pd.DataFrame(df_te)

df_all = pd.concat([df_tr, df_te]).reset_index(drop=True)

amt_list = []
for i, row in df_all.iterrows():
    if i % 1000 == 1:
        util.log('iter = %d' % i)
    amt = re.findall(u'(\d*\.?\d+)元', row['raw_content'].decode('utf8'))
    amt_tt = re.findall(u'(\d*\.?\d+)万元', row['raw_content'].decode('utf8'))
    for a in amt:
        amt_list.append([row['id'], float(a)])
    for a in amt_tt:
        amt_list.append([row['id'], float(a) * 10000])
amt_feat = pd.DataFrame(amt_list, columns=['id', 'amount'])
amt_feat = amt_feat.groupby('id')['amount'].agg([sum, min, max, np.ptp, np.mean, np.std]).reset_index()
amt_feat = pd.merge(df_all, amt_feat, how='left', on='id').drop(['id', 'raw_content'], axis=1)
amt_feat.columns = ['amt_' + i for i in amt_feat.columns]

amt_feat.to_csv(param.data_path + '/output/feature/amt/amt_21w.csv', index=None, encoding='utf8')
