import pickle  # 导入持久化类
from sklearn.datasets.base import Bunch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
import xgboost as xgb
from matplotlib import pyplot
import matplotlib.pylab as plt
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np


def load_bunch_obj(path):  # 读取bunch对象函数
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)  # 使用pickle.load反序列化对象
    file_obj.close()
    return bunch


def split_data_set(bunch):
    X = bunch.tdm
    y = bunch.label
    X, y = shuffle(X, y)

    split_ratio = 0.7
    split_index = int(len(y) * split_ratio)
    train_set = Bunch(target_name=bunch.target_name, label=y[:split_index], tdm=X[:split_index])
    test_set = Bunch(target_name=bunch.target_name, label=y[split_index:], tdm=X[split_index:])
    return train_set, test_set


def train_nb(X_train, y_train):
    # # alpha越小，迭代次数越多，精度越高
    # clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

    # 交叉验证
    model = MultinomialNB()
    alphas = [0.01, 0.001, 0.001]
    param_grid = dict(alpha=alphas)
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        scoring="neg_log_loss",
                        cv=kfold,
                        verbose=0
                        )
    clf = grid.fit(X_train, y_train)

    # Save trained model
    model_path = 'models/naive_bayes.pkl'
    save_model(model_path, clf)
    return clf


def train_random_forest(train_set):
    clf = RandomForestClassifier(max_depth=5)
    clf.fit(train_set.tdm, train_set.label)

    model_path = 'models/random_forest.pkl'
    save_model(model_path, clf)
    return clf


def train_sgd_clf(X_train, y_train):
    clf = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-4, max_iter=7, tol=None)
    clf.fit(X_train, y_train)

    # Save trained model
    model_path = 'models/sgd_clf.pkl'
    save_model(model_path, clf)
    return clf


def train_lsvc(X_train, y_train):
    clf = LinearSVC(loss='hinge')
    clf.fit(X_train, y_train)

    model_path = 'models/lsvc.pkl'
    save_model(model_path, clf)
    return clf


def modelfit2(alg, X_train, y_train, useTrainCV=True, cv_folds=None, early_stopping_rounds=50):
    # modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=None, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    # 得到最佳弱分类器的个数

    # alg.fit(X_train, y_train, eval_metric='auc')
    #
    # # Predict training set:
    # dtrain_predictions = alg.predict(dtrain[predictors])
    # dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    #
    # # Print model report:
    # print("\nModel Report")
    # print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


# 多分类交叉
def modelfit_MultiClassEvaluation(alg, X_train, y_train, useTrainCV=True, cv_folds=None, early_stopping_rounds=50):
    # 如果使用交叉验证
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 9

        # xgboost单独调用只支持DMatrix数据接口
        xgtrain = xgb.DMatrix(X_train, label=y_train)

        # 交叉验证
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds=cv_folds,
                          metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)

        # 得到最佳弱分类器的个数
        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators=n_estimators)

        print(cvresult)
        # result = pd.DataFrame(cvresult)   #cv缺省返回结果为DataFrame
        # result.to_csv('my_preds.csv', index_label = 'n_estimators')
        cvresult.to_csv('my_preds_4_1.csv', index_label='n_estimators')

        # plot
        test_means = cvresult['test-mlogloss-mean']
        test_stds = cvresult['test-mlogloss-std']

        train_means = cvresult['train-mlogloss-mean']
        train_stds = cvresult['train-mlogloss-std']

        x_axis = range(0, n_estimators)
        pyplot.errorbar(x_axis, test_means, yerr=test_stds, label='Test')
        pyplot.errorbar(x_axis, train_means, yerr=train_stds, label='Train')
        pyplot.title("XGBoost n_estimators vs Log Loss")
        pyplot.xlabel('n_estimators')
        pyplot.ylabel('Log Loss')
        pyplot.savefig('n_estimators.png')

    # 调用sklearn的接口。得到n_estimators，在用n_estimators训练。Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='mlogloss')

    # Predict training set:
    train_predprob = alg.predict_proba(X_train)
    logloss = log_loss(y_train, train_predprob)

    # Print model report:
    print("logloss of train :")
    print(logloss)


def modelfit_two_classification(alg, X_train, y_train, useTrainCV=True, cv_folds=None, early_stopping_rounds=50):
    # 如果使用交叉验证
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        # 多分类要设置分类
        # xgb_param['num_class'] = 2

        # xgboost单独调用只支持DMatrix数据接口,与sklearn下面的xgboost有些不一样
        xgtrain = xgb.DMatrix(X_train, label=y_train)

        # 交叉验证 ，多分类metrics='mlogloss'，二分类用auc
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)

        # 得到最佳弱分类器的个数
        n_estimators = cvresult.shape[0]
        print('best n_estimators=' + str(n_estimators))
        alg.set_params(n_estimators=n_estimators)
        print(cvresult)

        # ----额外功能，保存结果 begin--------#
        cvresult.to_csv('cross_validation_result/my_preds_4_1.csv', index_label='n_estimators')
        # ----额外功能，保存结果 begin--------#

    # 调用sklearn的接口。得到n_estimators，在用n_estimators训练。Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='mlogloss')

    # Predict training set:
    train_predprob = alg.predict_proba(X_train)
    logloss = log_loss(y_train, train_predprob)

    # Print model report:
    print("logloss of train :")
    print(logloss)


def optimize_n_estimators(xgb, X_train, y_train):
    print('参数调优...')
    # 各类样本不均衡（一般先要分析数据），交叉验证是采用StratifiedKFold，在每折采样时各类样本按比例采样
    # prepare cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    # 1 调整n_estimators
    print('optimize n_estimators...')
    modelfit_two_classification(xgb, X_train, y_train, cv_folds=kfold)
    print('optimize end!')


def optimize_gamma(xgb, X_train, y_train):
    print('参数调优...')
    # 各类样本不均衡（一般先要分析数据），交叉验证是采用StratifiedKFold，在每折采样时各类样本按比例采样
    # prepare cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    gamma = [i / 10.0 for i in range(0, 5)]

    param_test = dict(gamma=gamma)
    print(param_test)

    print('optimize gamma...')
    # scoring = 'roc_auc'
    # scoring = 'neg_log_loss'
    gsearch = GridSearchCV(xgb, param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=kfold)
    gsearch.fit(X_train, y_train)
    print('grid_scores:')
    print(gsearch.grid_scores_)
    print('best_params:')
    print(gsearch.best_params_)
    print('best_score:')
    print(gsearch.best_score_)
    print('调整结束！')


# 正则化参数调优
def optimize_reg_alpha(xgb, X_train, y_train):
    print('参数调优...')
    # 各类样本不均衡（一般先要分析数据），交叉验证是采用StratifiedKFold，在每折采样时各类样本按比例采样
    # prepare cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    reg_alpha = [1e-5, 1e-2, 0.1, 1, 100]
    param_test = dict(reg_alpha=reg_alpha)
    print(param_test)

    print('optimize gamma...')
    # scoring = 'roc_auc'
    # scoring = 'neg_log_loss'
    gsearch = GridSearchCV(xgb, param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=kfold)
    gsearch.fit(X_train, y_train)
    print('grid_scores:')
    print(gsearch.grid_scores_)
    print('best_params:')
    print(gsearch.best_params_)
    print('best_score:')
    print(gsearch.best_score_)
    print('调整结束！')


# 优化max_depth和min_child_weight
def optimize_maxDepth_minChildWeight(xgb, X_train, y_train):
    print('参数调优...')
    # 各类样本不均衡（一般先要分析数据），交叉验证是采用StratifiedKFold，在每折采样时各类样本按比例采样
    # prepare cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    # max_depth 建议3-10， min_child_weight=1／sqrt(ratio_rare_event) =5.5
    # 3到10，步长为2
    # -----粗调与细调 begin----------------#
    # 第一次设置，粗调
    # max_depth = range(3, 10, 2)
    # min_child_weight = range(1, 6, 2)

    # 第二次设置，细调
    # 在第一次设置基础上得到最佳后，再缩短步长 跑一遍
    # 根据上面的设置，得到最佳值 {'max_depth': 7, 'min_child_weight': 1}，则我们重新设置max_depth，min_child_weight
    # max_depth = [6, 7, 8]
    # min_child_weight = [1, 2, 3]

    # -----粗调与细调 end----------------#

    # 由于数据不多，所以我们不进行粗调，直接细调
    max_depth = range(3, 10)
    min_child_weight = range(1, 6)

    param_test2_1 = dict(max_depth=max_depth, min_child_weight=min_child_weight)
    print(param_test2_1)

    # max_depth=6与min_child_weight在这里 设置没关系，因为后面交叉验证会修改值
    print('optimize maxDepth minChildWeight...')
    # scoring = 'roc_auc'
    # scoring = 'neg_log_loss'
    gsearch = GridSearchCV(xgb, param_grid=param_test2_1, scoring='roc_auc', n_jobs=-1, cv=kfold)
    gsearch.fit(X_train, y_train)
    print('grid_scores:')
    print(gsearch.grid_scores_)
    print('best_params:')
    print(gsearch.best_params_)
    print('best_score:')
    print(gsearch.best_score_)

    print('调整结束！')


# 优化subsample和colsample_bytree
def optimize_subsample_colsample_bytree(xgb, X_train, y_train):
    print('参数调优...')
    # 各类样本不均衡（一般先要分析数据），交叉验证是采用StratifiedKFold，在每折采样时各类样本按比例采样
    # prepare cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    subsample = [i / 10.0 for i in range(6, 10)]
    colsample_bytree = [i / 10.0 for i in range(6, 10)]

    param_test = dict(subsample=subsample, colsample_bytree=colsample_bytree)
    print(param_test)

    print('subsample和colsample_bytree...')

    # scoring = 'roc_auc'
    # scoring = 'neg_log_loss'
    gsearch = GridSearchCV(xgb, param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=kfold)
    gsearch.fit(X_train, y_train)
    print('grid_scores:')
    print(gsearch.grid_scores_)
    print('best_params:')
    print(gsearch.best_params_)
    print('best_score:')
    print(gsearch.best_score_)

    print('调整结束！')


# 参数调优
def xgb_cross_validation(X_train, y_train):
    # 初始化模型
    xgb = XGBClassifier(
        n_estimators=1000,  # 树的个数
        silent=0,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        # nthread=4,# cpu 线程数 默认最大
        learning_rate=0.1,  # 学习率，以前为0.3
        min_child_weight=1,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=6,  # 构建树的深度，越大越容易过拟合
        gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=0.7,  # 随机采样训练样本 训练实例的子采样比
        colsample_bytree=0.6,  # 生成树时进行的列采样
        colsample_bylevel=0.5,
        max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        # reg_alpha=0, # L1 正则项参数
        # scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        # objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
        # num_class=2, # 类别数，多分类与 multisoftmax 并用
        objective='binary:logistic',
        seed=1000  # 随机种子
        # eval_metric= 'auc'
    )

    # step1.调整n_estimators
    # optimize_n_estimators(xgb, X_train, y_train)
    # n_estimators 在103时停止，由于从0开始，所以n_estimators最佳参数为104
    xgb.set_params(n_estimators=104)

    # step2.1.调整max_depth和min_child_weight.
    #  粗调，细调。细调整是在粗调整基础上的
    # optimize_maxDepth_minChildWeight(xgb, X_train, y_train)
    # 计算结果  best_params:{'min_child_weight': 2, 'max_depth': 4}
    xgb.set_params(max_depth=4, min_child_weight=2)

    # step2.2 完成step1、step2后，需要重新调整弱学习器的数量n_estimators，因为深度增加了，可能我们需要的树的数量会减少
    # optimize_n_estimators(xgb, X_train, y_train)
    # 计算结果：n_estimators 依然是104
    xgb.set_params(n_estimators=104)

    # step3.调整gamma,默认值都很好，方法与前面的一样
    # gamma：节点分裂所需的最小损失函数下降值
    # optimize_gamma(xgb, X_train, y_train)
    # best_params:{'gamma': 0.1}
    xgb.set_params(gamma=0.1)

    # step4.调整subsample和colsample_bytree
    # optimize_subsample_colsample_bytree(xgb, X_train, y_train)
    # best_params:{'colsample_bytree': 0.9, 'subsample': 0.9}
    xgb.set_params(colsample_bytree=0.9, subsample=0.9)

    # step5.调整正则化参数
    # optimize_reg_alpha(xgb, X_train, y_train)
    # best_params:{'reg_alpha': 0.1}
    xgb.set_params(reg_alpha=0.1)

    # step6.减小学习率
    # learning_rate 设置为0.01
    # xgb.set_params(learning_rate=0.01)

    # step7 完成优化所有参数，重新训练
    xgb.fit(X_train, y_train, eval_metric='auc')

    return xgb


def save_model(model_path, model):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def get_stopwords(path):
    return set([line.strip() for line in open(path, 'r', encoding='utf-8')])


def xgboost(X_train, y_train):
    model = xgb_cross_validation(X_train, y_train)

    # 保存模型
    model_path = 'data/models/xgboost.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model


# 根据句子tfidf作为输入
def train_by_tfidf():
    # 停用词
    stopwords_path = 'data/stop_words_ch.txt'
    trainset_path = "data/wordbag/trainset.dat"
    # vectorizer_pkl对象保存路径
    vectorizer_path = 'data/model/vectorizer.pkl'
    bunch_obj = load_bunch_obj(trainset_path)  # 导入训练集向量空间

    #####
    # 获取语料的分词结果
    X = bunch_obj.contents
    y = bunch_obj.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,
                                                        stratify=y)  ##test_size测试集合所占比例

    # 获取停用词
    stpwrdlst = get_stopwords(stopwords_path)

    # 使用TfidfVectorizer初始化向量空间模型--创建词袋
    vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, min_df=3)
    # 文本转为tf-idf权值矩阵
    weight_train = vectorizer.fit_transform(X_train)
    # 注意，上一个是fit_transform，这里只是transform
    weight_test = vectorizer.transform(X_test)

    # vocabulary = vectorizer.vocabulary_

    # vectorizer已经fit好，现在保存vectorizer,创建词袋的持久化
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print("if-idf词袋创建成功！")

    # train_set, test_set = split_data_set(bunch_obj)
    print('这里使用xgboost')

    user_input = input('\n输入模型类型：'
                       '\n1 - Naive Bayes'
                       '\n2 - Random Forest'
                       '\n3 - SGD Classifier'
                       '\n4 - Linear SVC'
                       '\n5 - XGBoost'
                       '\n6 - 退出'
                       '\n')

    if user_input == '6':
        return

    print('\nFitting model with training set...')
    if user_input == '1':
        model = train_nb(weight_train, y_train)
    elif user_input == '2':
        model = train_random_forest(weight_train, y_train)
    elif user_input == '3':
        model = train_sgd_clf(weight_train, y_train)
    elif user_input == '4':
        model = train_lsvc(weight_train, y_train)
    elif user_input == '5':
        model = xgboost(weight_train, y_train)
    print('\nFitting completed')

    # 预测分类结果

    y_pred = model.predict(weight_test)
    # 注意bunch_obj.target_name=['neg','pos'] ，由于label为target_name.index(mydir)，所以neg表示0类，pos表示1类
    print(metrics.classification_report(y_test, y_pred, target_names=bunch_obj.target_name))
    # print('accuracy: ' + str(metrics.accuracy_score(y_test, y_pred)))
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))


# 处理语料，把分词后的句子变为list，且加入停用词
def process_corpus(contents, stpwrdlst):
    ret = []
    for cont in contents:
        cont = str(cont)
        words = cont.split(' ')
        lst = []
        for word in words:
            word = word.strip()
            if word not in stpwrdlst and word != '':
                lst.append(word)
        ret.append(lst)
    return ret


# 对每个句子的所有词向量取均值
def buildWordVector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            aa = imdb_w2v[word].reshape((1, size))
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算词向量，且保存
def train_vecs_save(x_train, x_test, y_train, y_test, n_dim):
    # Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=3)
    imdb_w2v.build_vocab(x_train)

    # Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x_train)

    train_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in x_train])
    # train_vecs = scale(train_vecs)

    np.save('word2vec_data/train_vecs.npy', train_vecs)
    print(train_vecs.shape)
    # Train word2vec_data on test tweets
    imdb_w2v.train(x_test)
    imdb_w2v.save('word2vec_data/w2v_model/w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in x_test])
    # test_vecs = scale(test_vecs)
    np.save('word2vec_data/test_vecs.npy', test_vecs)

    np.save('word2vec_data/y_train.npy', y_train)
    np.save('word2vec_data/y_test.npy', y_test)
    print(test_vecs.shape)


def get_data():
    train_vecs = np.load('word2vec_data/train_vecs.npy')
    y_train = np.load('word2vec_data/y_train.npy')
    test_vecs = np.load('word2vec_data/test_vecs.npy')
    y_test = np.load('word2vec_data/y_test.npy')
    return train_vecs, y_train, test_vecs, y_test


# 对列表进行分词并用空格连接
def segmentWord(txt, stopwords_list):
    lst = []
    word_list = list(jieba.cut(txt))
    for word in word_list:
        word = word.strip()
        if word not in stopwords_list and word != '':
            lst.append(word)
    return lst


##得到待预测单个句子的词向量
def get_predict_vecs(words, n_dim):
    imdb_w2v = Word2Vec.load('word2vec_data/w2v_model/w2v_model.pkl')
    # imdb_w2v.train(words)
    train_vecs = buildWordVector(words, n_dim, imdb_w2v)
    # print train_vecs.shape
    return train_vecs


# 用gesim的Word2Vec作为输入
def train_by_word2vec():
    # 停用词
    stopwords_path = 'data/stop_words_ch.txt'
    trainset_path = 'data/wordbag/trainset.dat'
    bunch_obj = load_bunch_obj(trainset_path)  # 导入训练集向量空间

    #####
    # 获取停用词
    stpwrdlst = get_stopwords(stopwords_path)
    # 获取语料的分词结果
    contents = bunch_obj.contents

    X = process_corpus(contents, stpwrdlst)
    y = bunch_obj.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,
                                                        stratify=y)  ##test_size测试集合所占比例

    n_dim = 100
    ## 计算词向量，且保存
    train_vecs_save(X_train, X_test, y_train, y_test, n_dim)
    train_vecs, y_train, test_vecs, y_test = get_data()  # 导入训练数据和测试数据

    # -----------------predict---begin-----------------#
    # 补充，对于新的句子，我们可以求出他的vec
    # string = '我爱我的祖国，我的祖国很美好'
    # words = segmentWord(string)
    # #计算出预测的word2vec，可以用xgboost计算了
    # predict_vecs = get_predict_vecs(words, n_dim)
    # -----------------predict---end-----------------#


    # 模型参数优化
    model = XGBClassifier(
        n_estimators=105,  # 树的个数
        silent=0,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        # nthread=4,# cpu 线程数 默认最大
        learning_rate=0.05,  # 学习率，以前为0.3
        min_child_weight=2,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=7,  # 构建树的深度，越大越容易过拟合
        gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=0.7,  # 随机采样训练样本 训练实例的子采样比
        colsample_bytree=0.6,  # 生成树时进行的列采样
        colsample_bylevel=0.5,
        max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        # reg_alpha=0, # L1 正则项参数
        # scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        # objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
        # num_class=2, # 类别数，多分类与 multisoftmax 并用
        objective='binary:logistic',
        seed=1000  # 随机种子
        # eval_metric= 'auc'
    )
    model.fit(train_vecs, y_train, eval_metric='auc')

    # 预测分类结果
    y_pred = model.predict(test_vecs)
    # 注意bunch_obj.target_name=['neg','pos'] ，由于label为target_name.index(mydir)，所以neg表示0类，pos表示1类
    print(metrics.classification_report(y_test, y_pred, target_names=bunch_obj.target_name))
    # print('accuracy: ' + str(metrics.accuracy_score(y_test, y_pred)))
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))


# 主要是参数调优
def main():
    # 方法1：用tfidf 词频
    train_by_tfidf()

    # 方法2：用gesim的Word2Vec作为输入，由于训练数据太少，导致word2vec正确率很低，计划用doc2vec测试
    # n_dim = 100
    # train_by_word2vec()


if __name__ == "__main__":
    main()
