from __future__ import print_function

import warnings
import numpy as np
import pandas as pd
import os
import sys

from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.utils import shuffle, class_weight
from sklearn.preprocessing import StandardScaler
#from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import log_loss, make_scorer
from scipy.stats import uniform, randint
from sklearn import linear_model
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

global is_print, num_jobs

def showGridScore(fitObj):
    ''' print mean_test_score, std_test_score and corresponding parameters 
        Args:
            fitObj: the fitted GridSearchCV/RandomSearchCV object
    '''
    train_means = fitObj.cv_results_['mean_train_score']
    test_means = fitObj.cv_results_['mean_test_score']
    params = fitObj.cv_results_['params']
    
    for train, test, param in zip(train_means, test_means, params):
        print('%0.3f / %0.3f for %r' % (train, test, param))
    print('Best parameters and score: ', fitObj.best_params_, fitObj.best_score_, '\n')

def weighted_log_loss(y_true, y_pred, weight_dict):
    ''' compute weighted log loss using weight_dict 
    
    Args:
    y_pred y_true: n*1 or n*2 numpy array.
    weights_dict: python dictionary mapping class labels to their corresponding weights.
    '''
    
    sample_weights = np.repeat(weight_dict[0], y_true.shape[0])
    sample_weights[y_true == 1] = weight_dict[1]
    
    return log_loss(y_true, y_pred, sample_weight = sample_weights)

def getSearchPattern(loc, name = 'name'):
    '''
    loc the location of file, containing only one column for indicated drug list
    name a string specifies the name of the column
    return: a string concatenated by '|'
    '''
    drugList = pd.read_csv(loc, header=None)
    drugList.columns = [name]
    drugList = [d.lower() for d in drugList[name]]
    sePattern = "|".join(drugList)
    return sePattern

def getIndication(sePattern, drugList):
    ''' 
    use re package to find whether elements in drugList matches sePattern
    return: a list with 1 indicating the matching the pattern, otherwise not
    '''
    idx = map(lambda x: int(bool(re.search(sePattern, x, re.IGNORECASE))), drugList)
    return idx

def rbf_svm(cv_obj):

    # pipeline setup
    cls_svm = SVC(C=10.0, 
              kernel='rbf', 
              gamma=0.1, 
              # class_weight = weights,
              decision_function_shape='ovr', 
              probability=True)

    # kernel_svm = Pipeline([('std', StandardScaler()),  ('svc', cls)])

    # gridsearch setup
    # param_grid = [{
    #     'C': [1, 10, 100, 1000], 
    #     'gamma': [0.001, 0.0001], 
    #     'kernel': ['rbf']}
    #  ]
     
    # param_grid = [{
    #     'C': np.logspace(-2, 10, 13), 
    #     'gamma': np.logspace(-9, 3, 13), 
    #     'kernel': ['rbf']}
    # ]

    param_grid = [{
        'C': np.logspace(-5, 2, 8),
        'gamma': np.logspace(-6, 2, 5),
        'kernel': ['rbf']}
    ]

    gs_svm = GridSearchCV(estimator = cls_svm, 
                        param_grid = param_grid,
                        # fit_params={'sample_weight': train_weights},
                        scoring='neg_log_loss', 
                        n_jobs = num_jobs, 
                        cv=cv_obj, 
                        verbose=is_print, 
                        refit=True, 
                        pre_dispatch='2*n_jobs')

    return gs_svm

def random_forest(cv_obj):
    
    cls_RF = RandomForestClassifier(n_estimators=100, 
        criterion='gini', 
        max_depth=None, 
        max_features=100, 
        oob_score=True, 
        n_jobs = num_jobs, 
        random_state=None, 
        verbose=is_print 
    )
        
    # RF = Pipeline([('std', StandardScaler() ), ('rf', cls_RF)])
   
    param_grid_rf = {
        'n_estimators': [1000],  ##normally the more bagged trees the better, not likely to overfit 
        'max_depth': [None],   # max depth of trees, ESL book recommends full depth
        # 'min_samples_leaf': np.arange(10, 51, 10),
        'min_samples_leaf': [1, 3, 5, 10, 30, 50, 80],
        'max_features': [800, 1000, 1500, 2000, 3000, 5000]  ##default is square root of the no. of features (no. of features to condeir for each split), may set to higher if you believe there are few relevant features 
    }
    
    gs_rf = GridSearchCV(estimator=cls_RF, 
                    param_grid=param_grid_rf, 
                    # fit_params={'sample_weight': train_weights},
                    scoring='neg_log_loss',   #http://scikit-learn.org/stable/modules/model_evaluation.html
                    n_jobs = num_jobs, 
                    cv=cv_obj,
                    verbose=is_print, 
                    refit=True,
                    pre_dispatch='2*n_jobs')
    
    return gs_rf

def elastic_net(cv_obj):
    cls_enet = linear_model.SGDClassifier(loss="log", 
        penalty="elasticnet",
        alpha = 0.0001,
        l1_ratio = 0.5
    )
        
    enet = Pipeline([('std', StandardScaler()), ('enet', cls_enet)])

    param_grid_enet = {
        'enet__alpha': [0.01,0.005,1e-3 ,5e-4,1e-4,5e-5,1e-5], ##regularization strength 
        'enet__l1_ratio' : [0.25,0.5,0.75]  ##elastic net mixing parameter 
    }

    param_grid_enet_rand = {
        'enet__alpha': uniform(loc=1e-5, scale= 1e-2 - 1e-5), ##regularization strength 
        'enet__l1_ratio': uniform(loc=0, scale =1)  ##elastic net mixing parameter 
    }
    
    gs_enet_rand = RandomizedSearchCV(estimator = enet, 
            param_distributions = param_grid_enet_rand,
            scoring='neg_log_loss',
            n_jobs = num_jobs,
            n_iter=3,
            cv=cv_obj,
            verbose=is_print,  
            refit=True,
            pre_dispatch='2*n_jobs'
    )

    ## logistic regression with elastic net (ie glmnet in R)
    gs_enet = GridSearchCV(estimator=enet, 
                param_grid=param_grid_enet, 
                scoring='neg_log_loss',   #http://scikit-learn.org/stable/modules/model_evaluation.html
                n_jobs = num_jobs, 
                cv=cv_obj,
                verbose=is_print, 
                refit=True,
                pre_dispatch='2*n_jobs')
    
    return gs_enet_rand

def boosted_trees(cv_obj):
    cls_gbm = GradientBoostingClassifier(loss='deviance', 
        learning_rate=0.1,   ##high learning rate can lead to overfit
        n_estimators=100,   # no. of trees to fit, ie no. of iterations of boosting, normally increase in n_estimators won't overfit 
        subsample=0.8,       # subsampling of bservations to reduce overfitting
        max_depth=2,         # maximum depth of tree, too high may overfit   
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,   ##may need to set lower values for min_samples_split for unbalanced data
        max_features=None,   
        init=None, 
        random_state=10, 
        verbose=is_print, 
        presort='auto')

    # gbm = Pipeline([('std', StandardScaler()), ('gbm', cls_gbm)])

    # tuning parameters in gbm :   https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
    # gridsearch setup
    param_grid_gbm = {
        # 'learning_rate': [0.03, 0.05, 0.1, 0.2],  ##no. of iterations  not likely to overfit) 
        'learning_rate': [0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
        'n_estimators': range(100, 1001, 50), 
        'subsample': [1],  # uniform varaible from loc to (loc+scale)
        'max_depth': [2, 3, 5, 10],
        # 'min_samples_split': range(1, 50, 5),
        'max_features': [10, 30, 50, 100, 500, 1000] 
    }
        

    #random grid search
    param_grid_gbm_rand = {
        'learning_ate': uniform(loc=0.05,scale= 0.15),  ##no. of iterations  not likely to overfit) 
        'n_estimators': [100], 
        'subsample': uniform(loc=0.7, scale=0.3), # uniform varaible from loc to (loc+scale)
        'max_depth': randint(5,8),
        'min_samples_split': randint(2,10),
        'max_features':[10,20,30,50,100,500,1000,3000,5000] #randint(50,5000) 
    }
    
    ## boosted trees
    gs_gbm = GridSearchCV(estimator=cls_gbm, 
                    param_grid = param_grid_gbm, 
                    # fit_params={'sample_weight': train_weights},
                    scoring='neg_log_loss',   #http://scikit-learn.org/stable/modules/model_evaluation.html
                    # fit_params = params,
                    n_jobs = num_jobs, 
                    cv = cv_obj,
                    verbose=is_print, 
                    refit=True,
                    pre_dispatch='2*n_jobs')
                    
    gs_gbm_rand = RandomizedSearchCV(estimator = cls_gbm,
                param_distributions = param_grid_gbm_rand,
                # fit_params={'sample_weight': train_weights},
                scoring='neg_log_loss',
                # fit_params = params,
                n_jobs = num_jobs,
                n_iter = 20,
                cv = cv_obj, 
                verbose=is_print,  
                refit=True,
                pre_dispatch='2*n_jobs') 
    return gs_gbm

def write_file(file_name, drug_name, predict_result, real_result):
    assert len(drug_name) == len(predict_result) and len(predict_result) == len(real_result), "The length of arrays is not the same!" 
    
    with open(file_name, 'a') as f:
        for i in range(len(drug_name)):
            f.write('%s, %s, %s\n' % (drug_name[i], predict_result[i], real_result[i]))
        
    
if __name__ == '__main__':
    
    is_print = 0
    num_jobs = 6 
    data_location_dic = {
        'scz':"/exeh_3/kai/data/SCZ_indication_orig_drug_expr_with_DrugNames_standardized.csv",
        'antipsycho':"/exeh_3/kai/data/Cmap_differential_expression_antipsycho.csv",
        'antidepression':"/exeh_3/kai/data/Cmap_differential_expression_antidepression.csv",
        'depressionANDanxiety':"/exeh_3/kai/data/Cmap_differential_expression_anxiety_depression.csv"
    }
    data_source = 'depressionANDanxiety'

    os.chdir("/exeh_3/kai/GE_result/%s" % data_source) 

    os.system('touch -f svm_%s_weighted_result.out && > svm_%s_weighted_result.out' % (data_source, data_source))
    os.system('touch -f rf_%s_weighted_result.out && > rf_%s_weighted_result.out' % (data_source, data_source))
    os.system('touch -f bt_%s_weighted_result.out && > bt_%s_weighted_result.out' % (data_source, data_source))
    
    pheno = pd.read_csv(data_location_dic[data_source])
    X_orig = pheno.iloc[:, 2:].values #.values would transfer dataframe to an array; iloc is a function to index row/columns in dataframe
    y_orig = pheno.iloc[:, 1].values
    drug_name = pheno.iloc[:, 0].values
    
    # generate sample weight array
    class_weights = class_weight.compute_class_weight(classes = np.unique(y_orig), y = y_orig, class_weight = 'balanced')
    weights = np.repeat(class_weights[0], y_orig.shape[0])
    weights[y_orig == 1] = class_weights[1]
    print('class weights: %s.' % class_weights)
   
    # define the scoring method for grid search
    # partial_log_loss = partial(weighted_log_loss, weight_dict = weights)
    # w_log_loss = make_scorer(partial_log_loss, greater_is_better = False, needs_proba = True)

    svm_loss, rf_loss, bt_loss = [], [], []
    weighted_svm_loss, weighted_net_loss, weighted_rf_loss, weighted_bt_loss = [], [], [], []
    svm_result, net_result, rf_result, bt_result = None, None, None, None
    
    X_orig, y_orig, weights, drug_name = shuffle(X_orig, y_orig, weights, drug_name, random_state = 0)
    testFold = StratifiedKFold(n_splits = 3)
    trainFold = StratifiedKFold(n_splits = 3)
    
    for train_index, test_index in testFold.split(X_orig, y_orig):
        X_train, y_train = X_orig[train_index], y_orig[train_index]
        X_test, y_test = X_orig[test_index], y_orig[test_index]
        train_weights, test_weights = weights[train_index], weights[test_index]
        drug_name_test = drug_name[test_index]
        
        # standardize before cross validation
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
        # create models
        gs_svm = rbf_svm(trainFold)
        gs_bt = boosted_trees(trainFold)
        gs_rf = random_forest(trainFold)
        
        # fit models at training data
        gs_svm.fit(X_train, y_train, sample_weight = train_weights)
        gs_rf.fit(X_train, y_train, sample_weight = train_weights)
        gs_bt.fit(X_train, y_train, sample_weight = train_weights)
        
        # print fit result
        showGridScore(gs_svm)
        showGridScore(gs_rf)
        showGridScore(gs_bt)
        
        # save unweighed log_loss from various models in test set 
        svm_loss.append(gs_svm.score(X_test, y_test))
        rf_loss.append(gs_rf.score(X_test, y_test))
        bt_loss.append(gs_bt.score(X_test, y_test))
        
        # predict on the test set
        svm_res = (gs_svm.predict_proba(X_test))[:, 1]
        rf_res = (gs_rf.predict_proba(X_test))[:, 1]
        bt_res = (gs_bt.predict_proba(X_test))[:, 1]
        
        # compute weighted log loss
        # weighted_svm_loss.append(log_loss(y_test, svm_res, sample_weight = test_weights))
        # weighted_rf_loss.append(log_loss(y_test, rf_res, sample_weight = test_weights))
        # weighted_bt_loss.append(log_loss(y_test, bt_res, sample_weight = test_weights))

        # write predict results to files
        write_file('svm_%s_weighted_result.out' % data_source, drug_name_test, svm_res, y_test)
        write_file('rf_%s_weighted_result.out' % data_source, drug_name_test, rf_res, y_test)
        write_file('bt_%s_weighted_result.out' % data_source, drug_name_test, bt_res, y_test)

    # print('The weighted neg log loss for SVM: %s, average loss: %s' % (weighted_svm_loss, np.mean(weighted_svm_loss)))
    # print('The weighted neg log loss for Random Forest: %s, average loss: %s' % (weighted_rf_loss, np.mean(weighted_rf_loss)))
    # print('The weighted neg log loss for Boosted Trees: %s, average loss: %s' % (weighted_bt_loss, np.mean(weighted_bt_loss)))
    print('The neg log loss for SVM: %s, average loss: %s' % (svm_loss, np.mean(svm_loss)))
    print('The neg log loss for Random Forest: %s, average loss: %s' % (rf_loss, np.mean(rf_loss)))
    print('The neg log loss for Boosted Trees: %s, average loss: %s' % (bt_loss, np.mean(bt_loss)))
