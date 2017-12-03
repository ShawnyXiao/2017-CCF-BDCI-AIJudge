# coding=utf-8
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score

import param
import util


############################ 定义评估函数 ############################
def micro_avg_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


############################ 加载数据 ############################
df_all = pd.read_csv(param.data_path + '/output/corpus/all_data.csv', encoding='utf8', nrows=param.train_num)
df_all['penalty'] = df_all['penalty'] - 1

model = Doc2Vec.load(param.data_path + '/output/model/dm_d2v_12w.model')
x_sp = np.array([model.docvecs[i] for i in range(param.train_num)])

############################ dmd2v stack ############################
np.random.seed(param.seed) # 固定种子，方便复现
df_stack = pd.DataFrame(index=range(len(df_all)))
tr_num = param.cv_train_num
num_class = len(pd.value_counts(df_all['penalty']))
n = 5

x = x_sp[:tr_num]
y = df_all['penalty'][:tr_num]
x_te = x_sp[tr_num:]
y_te = df_all['penalty'][tr_num:]

feat = 'dmd2v'
stack = np.zeros((x.shape[0], num_class))
stack_te = np.zeros((x_te.shape[0], num_class))

score_va = 0
score_te = 0
for i, (tr, va) in enumerate(StratifiedKFold(y, n_folds=n, random_state=param.seed)):
    util.log('stack:%d/%d' % ((i + 1), n))
    y_train = np_utils.to_categorical(y[tr], num_class)
    y_test = np_utils.to_categorical(y_te, num_class)
    model = Sequential()
    model.add(Dense(300, input_shape=(x[tr].shape[1],)))
    model.add(Dropout(0.1))
    model.add(Activation('tanh'))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    history = model.fit(x[tr], y_train, shuffle=True,
                        batch_size=128, nb_epoch=35,
                        verbose=2, validation_data=(x_te, y_test))
    y_pred_va = model.predict_proba(x[va])
    y_pred_te = model.predict_proba(x_te)
    util.log('va acc:%f' % micro_avg_f1(y[va], model.predict_classes(x[va])))
    util.log('te acc:%f' % micro_avg_f1(y_te, model.predict_classes(x_te)))
    score_va += micro_avg_f1(y[va], model.predict_classes(x[va]))
    score_te += micro_avg_f1(y_te, model.predict_classes(x_te))
    stack[va] += y_pred_va
    stack_te += y_pred_te
score_va /= n
score_te /= n
util.log('va avg acc:%f' % score_va)
util.log('te avg acc:%f' % score_te)
stack_te /= n
stack_all = np.vstack([stack, stack_te])
for l in range(stack_all.shape[1]):
    df_stack['{}_{}'.format(feat, l)] = stack_all[:, l]

df_stack.to_csv(param.data_path + '/output/feature/dmd2v/nn_prob_12w.csv', encoding='utf8', index=None)
util.log('Save dmd2v stack done!')