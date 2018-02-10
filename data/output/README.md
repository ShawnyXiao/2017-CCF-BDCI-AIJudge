# 输出数据

由于文件太大，因此无法上传到 Github 上，若需要相关数据，可以发 issue 联系我。该目录结构如下：

```
output
│  README.md
│
├─corpus
│      all_data.csv
│      doc_for_d2v_12w.txt
│      doc_for_d2v_21w.txt
│
├─feature
│  ├─amt
│  │      amt_12w.csv
│  │      amt_21w.csv
│  │
│  ├─dbowd2v
│  │      nn_prob_12w.csv
│  │      nn_prob_21w.csv
│  │
│  ├─dmd2v
│  │      nn_prob_12w.csv
│  │      nn_prob_21w.csv
│  │
│  ├─tfidf
│  │      bnb_prob_12w.csv
│  │      bnb_prob_21w.csv
│  │      lr_prob_12w.csv
│  │      lr_prob_21w.csv
│  │      mnb_prob_12w.csv
│  │      mnb_prob_21w.csv
│  │      svc_prob_12w.csv
│  │      svc_prob_21w.csv
│  │
│  └─w2v
│          w2v_12w.csv
│          w2v_avg_12w.csv
│
├─model
│      dbow_d2v_12w.model
│      dbow_d2v_12w.model.docvecs.doctag_syn0.npy
│      dbow_d2v_12w.model.syn1neg.npy
│      dbow_d2v_12w.model.wv.syn0.npy
│      dbow_d2v_21w.model
│      dbow_d2v_21w.model.docvecs.doctag_syn0.npy
│      dbow_d2v_21w.model.syn1neg.npy
│      dbow_d2v_21w.model.wv.syn0.npy
│      dm_d2v_12w.model
│      dm_d2v_12w.model.docvecs.doctag_syn0.npy
│      dm_d2v_12w.model.syn1neg.npy
│      dm_d2v_12w.model.wv.syn0.npy
│      dm_d2v_21w.model
│      dm_d2v_21w.model.docvecs.doctag_syn0.npy
│      dm_d2v_21w.model.syn1neg.npy
│      dm_d2v_21w.model.wv.syn0.npy
│      w2v_12w.model
│      w2v_12w.model.syn1neg.npy
│      w2v_12w.model.wv.syn0.npy
│
└─result
    ├─sub
    │      1123-xgb-tfidf_lr-r121.json
    │      1124-xgb-tfidf_lr_bnb_mnb+amt-r410.json
    │      1124-xgb-tfidf_lr_bnb_mnb-r372.json
    │      1201-xgb-tfidf_lr_bnb_mnb+amt+dbowd2v_nn-r518.json
    │      1209-xgb-tfidf_lr_bnb_mnb+amt+dbowd2v_nn+w2v-r1086.json
    │
    └─val
            1123-xgb-tfidf_lr(0.460300).json
            1123-xgb-tfidf_lr_bnb(0.460000).json
            1123-xgb-tfidf_lr_mnb(0.458150).json
            1124-xgb-tfidf_lr_bnb_mnb(0.460900).json
            1124-xgb-tfidf_lr_bnb_mnb+amt(0.479800).json
            1124-xgb-tfidf_lr_bnb_mnb_svc(0.458700).json
            1124-xgb-tfidf_lr_bnb_svc(0.458800).json
            1124-xgb-tfidf_lr_svc(0.459400).json
            1124-xgb-tfidf_svc(0.454650).json
            1201-xgb-tfidf_lr_bnb_mnb+amt+dbow_nn(0.481300).json
            1209-xgb-tfidf_lr_bnb_mnb+amt+dbow_nn+w2v(0.49375).json
            1210-xgb-tfidf_lr_bnb_mnb+amt+dbow_nn+dm_nn+w2v(0.49155).json
```