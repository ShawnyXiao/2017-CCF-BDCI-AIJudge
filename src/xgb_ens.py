# coding=utf-8
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

import param


############################ 定义评估函数 ############################
def micro_avg_f1(preds, dtrain):
    y_true = dtrain.get_label()
    return 'micro_avg_f1', f1_score(y_true, preds, average='micro')


############################ 加载特征 & 标签 ############################
df_tfidf_lr = pd.read_csv(param.data_path + '/output/feature/tfidf/lr_prob_12w.csv')
df_tfidf_bnb = pd.read_csv(param.data_path + '/output/feature/tfidf/bnb_prob_12w.csv')
df_tfidf_mnb = pd.read_csv(param.data_path + '/output/feature/tfidf/mnb_prob_12w.csv')
df_tfidf_svc = pd.read_csv(param.data_path + '/output/feature/tfidf/svc_prob_12w.csv')
df_amt = pd.read_csv(param.data_path + '/output/feature/amt/amt_12w.csv')
df_dbow_nn = pd.read_csv(param.data_path + '/output/feature/dbowd2v/nn_prob_12w.csv')
df_w2v = pd.read_csv(param.data_path + '/output/feature/w2v/w2v_12w.csv')
# df_dm = pd.read_csv(param.data_path + 'dmd2v_stack_20W.csv')

df_lb = pd.read_csv(param.data_path + '/output/corpus/all_data.csv', usecols=['id', 'penalty'], nrows=param.train_num)
df_lb['penalty'] = df_lb['penalty'] - 1  # 让标签属于 [0, 8)

############################ xgboost ############################
tr_num = param.cv_train_num
df_sub = pd.DataFrame()
df_sub['id'] = df_lb.iloc[tr_num:]['id']
seed = param.seed

n_trees = 10000
esr = 200
evals = 20

df = pd.concat([df_tfidf_lr, df_tfidf_bnb, df_tfidf_mnb, df_amt, df_dbow_nn, df_w2v], axis=1)
print(df.columns)
num_class = len(pd.value_counts(df_lb['penalty']))
x = df.iloc[:tr_num]
y = df_lb['penalty'][:tr_num]
x_te = df.iloc[tr_num:]
y_te = df_lb['penalty'][tr_num:]

max_depth = 7
min_child_weight = 1
subsample = 0.8
colsample_bytree = 0.8
gamma = 1
lam = 0

params = {
    'objective': 'multi:softmax',
    'booster': 'gbtree',
    'stratified': True,
    'num_class': num_class,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree,
    #     'gamma': gamma,
    #     'lambda': lam,

    'eta': 0.02,
    'silent': 1,
    'seed': seed,
}

dtrain = xgb.DMatrix(x, y)
dvalid = xgb.DMatrix(x_te, y_te)
watchlist = [(dtrain, 'train'), (dvalid, 'test')]
bst = xgb.train(params, dtrain, n_trees, evals=watchlist, feval=micro_avg_f1, maximize=True,
                early_stopping_rounds=esr, verbose_eval=evals)
df_sub['penalty'] = (bst.predict(dvalid) + 1).astype(int)

df_sub['id'] = df_sub['id'].astype(str)
df_sub['laws'] = [[1]] * len(df_sub)
df_sub.to_json(param.data_path + '/output/result/val/1209-xgb-tfidf_lr_bnb_mnb+amt+dbow_nn+w2v.json', orient='records', lines=True)
