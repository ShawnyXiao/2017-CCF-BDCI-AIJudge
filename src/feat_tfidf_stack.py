# coding=utf-8
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

import param
import util


############################ 定义评估函数 ############################
def micro_avg_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


############################ 加载数据 ############################
df_all = pd.read_csv(param.data_path + '/output/corpus/all_data.csv', encoding='utf8', nrows=param.train_num)
df_all['penalty'] = df_all['penalty'] - 1

############################ tfidf ############################
tfv = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf=True)
x_sp = tfv.fit_transform(df_all['content'])

############################ lr stack ############################
tr_num = param.cv_train_num
num_class = len(pd.value_counts(df_all['penalty']))
n = 5

x = x_sp[:tr_num]
y = df_all['penalty'][:tr_num]
x_te = x_sp[tr_num:]
y_te = df_all['penalty'][tr_num:]

stack = np.zeros((x.shape[0], num_class))
stack_te = np.zeros((x_te.shape[0], num_class))

score_va = 0
score_te = 0
for i, (tr, va) in enumerate(StratifiedKFold(y, n_folds=n, random_state=param.seed)):
    util.log('stack:%d/%d' % ((i + 1), n))
    clf = LogisticRegression(C=2)
    clf.fit(x[tr], y[tr])
    y_pred_va = clf.predict_proba(x[va])
    y_pred_te = clf.predict_proba(x_te)
    util.log('va acc:%f' % micro_avg_f1(y[va], clf.predict(x[va])))
    util.log('te acc:%f' % micro_avg_f1(y_te, clf.predict(x_te)))
    score_va += micro_avg_f1(y[va], clf.predict(x[va]))
    score_te += micro_avg_f1(y_te, clf.predict(x_te))
    stack[va] += y_pred_va
    stack_te += y_pred_te
score_va /= n
score_te /= n
util.log('va avg acc:%f' % score_va)
util.log('te avg acc:%f' % score_te)
stack_te /= n
stack_all = np.vstack([stack, stack_te])
df_stack = pd.DataFrame(index=range(len(df_all)))
for i in range(stack_all.shape[1]):
    df_stack['tfidf_lr_{}'.format(i)] = stack_all[:, i]

df_stack.to_csv(param.data_path + '/output/feature/tfidf/lr_prob_12w.csv', index=None, encoding='utf8')

############################ bnb stack ############################
tr_num = param.cv_train_num
num_class = len(pd.value_counts(df_all['penalty']))
n = 5

x = x_sp[:tr_num]
y = df_all['penalty'][:tr_num]
x_te = x_sp[tr_num:]
y_te = df_all['penalty'][tr_num:]

stack = np.zeros((x.shape[0], num_class))
stack_te = np.zeros((x_te.shape[0], num_class))

score_va = 0
score_te = 0
for i, (tr, va) in enumerate(StratifiedKFold(y, n_folds=n, random_state=param.seed)):
    util.log('stack:%d/%d' % ((i + 1), n))
    clf = BernoulliNB()
    clf.fit(x[tr], y[tr])
    y_pred_va = clf.predict_proba(x[va])
    y_pred_te = clf.predict_proba(x_te)
    util.log('va acc:%f' % micro_avg_f1(y[va], clf.predict(x[va])))
    util.log('te acc:%f' % micro_avg_f1(y_te, clf.predict(x_te)))
    score_va += micro_avg_f1(y[va], clf.predict(x[va]))
    score_te += micro_avg_f1(y_te, clf.predict(x_te))
    stack[va] += y_pred_va
    stack_te += y_pred_te
score_va /= n
score_te /= n
util.log('va avg acc:%f' % score_va)
util.log('te avg acc:%f' % score_te)
stack_te /= n
stack_all = np.vstack([stack, stack_te])
df_stack = pd.DataFrame(index=range(len(df_all)))
for i in range(stack_all.shape[1]):
    df_stack['tfidf_bnb_{}'.format(i)] = stack_all[:, i]

df_stack.to_csv(param.data_path + '/output/feature/tfidf/bnb_prob_12w.csv', index=None, encoding='utf8')

############################ mnb stack ############################
tr_num = param.cv_train_num
num_class = len(pd.value_counts(df_all['penalty']))
n = 5

x = x_sp[:tr_num]
y = df_all['penalty'][:tr_num]
x_te = x_sp[tr_num:]
y_te = df_all['penalty'][tr_num:]

stack = np.zeros((x.shape[0], num_class))
stack_te = np.zeros((x_te.shape[0], num_class))

score_va = 0
score_te = 0
for i, (tr, va) in enumerate(StratifiedKFold(y, n_folds=n, random_state=param.seed)):
    util.log('stack:%d/%d' % ((i + 1), n))
    clf = MultinomialNB()
    clf.fit(x[tr], y[tr])
    y_pred_va = clf.predict_proba(x[va])
    y_pred_te = clf.predict_proba(x_te)
    util.log('va acc:%f' % micro_avg_f1(y[va], clf.predict(x[va])))
    util.log('te acc:%f' % micro_avg_f1(y_te, clf.predict(x_te)))
    score_va += micro_avg_f1(y[va], clf.predict(x[va]))
    score_te += micro_avg_f1(y_te, clf.predict(x_te))
    stack[va] += y_pred_va
    stack_te += y_pred_te
score_va /= n
score_te /= n
util.log('va avg acc:%f' % score_va)
util.log('te avg acc:%f' % score_te)
stack_te /= n
stack_all = np.vstack([stack, stack_te])
df_stack = pd.DataFrame(index=range(len(df_all)))
for i in range(stack_all.shape[1]):
    df_stack['tfidf_gnb_{}'.format(i)] = stack_all[:, i]

df_stack.to_csv(param.data_path + '/output/feature/tfidf/gnb_prob_12w.csv', index=None, encoding='utf8')

############################ svc stack ############################
tr_num = param.cv_train_num
num_class = len(pd.value_counts(df_all['penalty']))
n = 5

x = x_sp[:tr_num]
y = df_all['penalty'][:tr_num]
x_te = x_sp[tr_num:]
y_te = df_all['penalty'][tr_num:]

stack = np.zeros((x.shape[0], num_class))
stack_te = np.zeros((x_te.shape[0], num_class))

score_va = 0
score_te = 0
for i, (tr, va) in enumerate(StratifiedKFold(y, n_folds=n, random_state=param.seed)):
    util.log('stack:%d/%d' % ((i + 1), n))
    clf = svm.LinearSVC(loss='hinge', tol=0.000001, C=0.5, verbose=1, random_state=param.seed, max_iter=5000)
    clf.fit(x[tr], y[tr])
    y_pred_va = clf.decision_function(x[va])
    y_pred_te = clf.decision_function(x_te)
    util.log('va acc:%f' % micro_avg_f1(y[va], clf.predict(x[va])))
    util.log('te acc:%f' % micro_avg_f1(y_te, clf.predict(x_te)))
    score_va += micro_avg_f1(y[va], clf.predict(x[va]))
    score_te += micro_avg_f1(y_te, clf.predict(x_te))
    stack[va] += y_pred_va
    stack_te += y_pred_te
score_va /= n
score_te /= n
util.log('va avg acc:%f' % score_va)
util.log('te avg acc:%f' % score_te)
stack_te /= n
stack_all = np.vstack([stack, stack_te])
df_stack = pd.DataFrame(index=range(len(df_all)))
for i in range(stack_all.shape[1]):
    df_stack['tfidf_svc_{}'.format(i)] = stack_all[:, i]

df_stack.to_csv(param.data_path + '/output/feature/tfidf/svc_prob_12w.csv', index=None, encoding='utf8')
