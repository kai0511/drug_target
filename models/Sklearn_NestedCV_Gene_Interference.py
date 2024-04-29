from __future__ import print_function

# import warnings
import numpy as np
import pandas as pd
import os
import re
import sys
import mygene

# from functools import partial
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.utils import shuffle, class_weight
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import log_loss, average_precision_score, roc_auc_score #, make_scorer
#from scipy.stats import uniform, randint
# from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
    drugList = pd.read_table(loc, header=None)
    drugList.columns = [name]
    drugList = [d.strip().lower() for d in drugList[name]]
    sePattern = "|".join(drugList)
    return sePattern

def getIndication(sePattern, drugList):
    ''' 
    use re package to find whether elements in drugList matches sePattern
    return: a list with 1 indicating the matching the pattern, otherwise not
    '''
    idx = list(map(lambda x: int(bool(re.search(sePattern, x, re.IGNORECASE))), drugList))
    return idx

def predict_and_save(fitted_obj, pert, estimator_name, pertType, ind, idx=1):
    mg = mygene.MyGeneInfo()
    pertRes = fitted_obj.predict_proba(pert.iloc[:, 1:].values)
    geneNames = [(g['entrezgene'], g['symbol']) for g in mg.getgenes(pert['perturbagen'].values)]
    nameList = list(zip(*geneNames))
    resDict = {'Id': pert.loc[:, 'perturbagen'], 'Name': nameList[1], "predRes": pd.Series(pertRes[:,idx])}
    pred_res = pd.DataFrame.from_dict(resDict)
    pred_res.to_csv(ind + '/' + estimator_name + '_' + pertType + '_' + ind + '_Res.csv', index = False)
    
def printEvaluationRes(resList, evalList, model):
    assert len(resList) == len(evalList), \
        print('The length of evalList and resList is not the same!')
    for i in range(len(resList)):
        print('%s for %s: %s, average loss: %s' % (evalList[i], model, resList[i], np.mean(resList[i])))
    
def write_file(file_name, drug_name, predict_result, real_result):
    assert len(drug_name) == len(predict_result) and len(predict_result) == len(real_result), "The length of arrays is not the same!" 
    
    with open(file_name, 'a') as f:
        for i in range(len(drug_name)):
            f.write('%s, %s, %s\n' % (drug_name[i], predict_result[i], real_result[i]))
        
    
def rbf_svm(cv_obj):

    # pipeline setup
    cls_svm = SVC(C=10.0, 
              kernel='rbf', 
              gamma=0.1, 
              # class_weight = weights,
              decision_function_shape='ovr', 
              probability=True)

    param_grid = [
        {'C': np.logspace(-1, 3, num = 19, base = 2),  # for diabetes: np.arange(600, 901, 50)
        'gamma': np.logspace(-20, -12, num = 19, base = 2),  # for diabetes: np.logspace(-7, -4, 50)
        'kernel': ['rbf']}
    ]

    gs_svm = GridSearchCV(estimator = cls_svm, 
                        param_grid = param_grid,
                        # fit_params={'sample_weight': train_weights},
                        scoring = 'neg_log_loss', 
                        n_jobs = num_jobs, 
                        cv = cv_obj, 
                        verbose = is_print, 
                        refit = True, 
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
        'learning_rate': [0.002, 0.005, 0.008, 0.01],
        'n_estimators': range(300, 701, 50), 
        'subsample': [1],  # uniform varaible from loc to (loc+scale)
        'max_depth': [8, 10, 15],
        # 'min_samples_split': range(1, 50, 5),
        'max_features': [30, 50, 100, 150, 200, 250] 
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
    return gs_gbm


if __name__ == '__main__':
    nfold, num_jobs, is_print = 5, 60, 0 
    disease = 'arthritis'
    indication_dict =  {'antidepression':'/exeh_3/kai/data/backups/ATC_antidepression.csv',
                        'antipsychotics':'/exeh_3/kai/data/backups/ATC_antipsychotics.csv',
                        'anxiety_depression':'/exeh_3/kai/data/backups/MEDI_anxiety_depression.csv',
                        'diabetes': '/exeh_3/kai/data/backups/ATC_diabetes.csv',
                        'hypertension': '/exeh_3/kai/data/backups/ATC_hypertension.csv',
                        'arthritis': '/exeh_3/kai/data/backups/MEDI_Rheumatoid_arthritis.csv',
                        'asthma': '/exeh_3/kai/data/backups/ATC_asthma.csv',
                        'scz':'/exeh_3/kai/data/backups/MEDI_scz.csv'}
    
    predictionData = {'knockdown': 'consensi-knockdown.tsv',
                      'overexpression':'consensi-overexpression.tsv'}
                      # 'pert_id':'consensi-pert_id.tsv'}
    
    os.chdir('/exeh_3/kai/GE_result/genePerturbation')
    pheno = pd.read_csv('cmap_drug_expression_profile.csv', header = 0)

    # generate indication
    sePattern = getSearchPattern(indication_dict[disease])
    indication = getIndication(sePattern, pheno['drugName'])
    
    X_orig = pheno.iloc[:, 2:].values
    y_orig = np.asarray(indication)
    drug_name = pheno.iloc[:, 0].values
    print('number of positive obs.: %s.' % np.sum(y_orig))

    # generate sample weight array
    class_weights = class_weight.compute_class_weight(classes = np.unique(y_orig), y = y_orig, class_weight = 'balanced')
    weights = np.repeat(class_weights[0], y_orig.shape[0])
    weights[y_orig == 1] = class_weights[1]
    print('class weights: %s.' % class_weights)
    
    svm_loss, rf_loss, bt_loss = [], [], []
    svm_roc, rf_roc, bt_roc = [], [], []
    svm_prc, rf_prc, bt_prc = [], [], []
    svm_mdl, rf_mdl, bt_mdl = [], [], []
    
    X_orig, y_orig, weights, drug_name = shuffle(X_orig, y_orig, weights, drug_name, random_state = 0)
    testFold = StratifiedKFold(n_splits = nfold)
    trainFold = StratifiedKFold(n_splits = nfold)
    
    for train_index, test_index in testFold.split(X_orig, y_orig):
        X_train, y_train = X_orig[train_index], y_orig[train_index]
        X_test, y_test = X_orig[test_index], y_orig[test_index]
        train_weights, test_weights = weights[train_index], weights[test_index]
        drug_name_test = drug_name[test_index]
        
        # standardize before cross validation
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)
    
        # create models
        # gs_svm = rbf_svm(trainFold)
        # gs_bt = boosted_trees(trainFold)
        gs_rf = random_forest(trainFold)
        
        # fit models at training data
        # gs_svm.fit(X_train, y_train, sample_weight = train_weights)
        gs_rf.fit(X_train, y_train, sample_weight = train_weights)
        # gs_bt.fit(X_train, y_train, sample_weight = train_weights)
        
        # print fit result
        # showGridScore(gs_svm)
        showGridScore(gs_rf)
        # showGridScore(gs_bt)
        
        # save unweighed log_loss from various models in test set 
        # svm_loss.append(gs_svm.score(X_test, y_test))
        rf_loss.append(gs_rf.score(X_test, y_test))
        # bt_loss.append(gs_bt.score(X_test, y_test))
        
        # save the best model from each CV
        # svm_mdl.append(gs_svm)
        rf_mdl.append(gs_rf)
        # bt_mdl.append(gs_bt)
        
        # predict on the test set
        # svm_pred = (gs_svm.predict_proba(X_test))[:, 1]
        rf_pred = (gs_rf.predict_proba(X_test))[:, 1]
        # bt_pred = (gs_bt.predict_proba(X_test))[:, 1]
        
        # combine and save results
        # svm_result = pd.DataFrame({'name': drug_name_test, 'y': y_test, 'pred': svm_pred}, columns=['name', 'y', 'pred'])
        rf_result = pd.DataFrame({'name': drug_name_test, 'y': y_test, 'pred': rf_pred}, columns=['name', 'y', 'pred'])
        # bt_result = pd.DataFrame({'name': drug_name_test, 'y': y_test, 'pred': bt_pred}, columns=['name', 'y', 'pred'])
        # svm_result.to_csv(disease + '/svm_' + disease + '_result.csv', index = False, mode = 'a')
        rf_result.to_csv(disease + '/rf_' + disease + '_result.csv', index = False, mode = 'a')
        # bt_result.to_csv(disease + '/bt_' + disease + '_result.csv', index = False, mode = 'a')
        
        # compute roc 
        # svm_roc.append(roc_auc_score(y_test, svm_pred))
        rf_roc.append(roc_auc_score(y_test, rf_pred))
        # bt_roc.append(roc_auc_score(y_test, bt_pred))
        
        # compute prc
        # svm_prc.append(average_precision_score(y_test, svm_pred))
        rf_prc.append(average_precision_score(y_test, rf_pred))
        # bt_prc.append(average_precision_score(y_test, bt_pred))
    
    # printEvaluationRes([svm_loss, svm_roc, svm_prc], ['LogLoss', 'AUC-ROC', 'AUC-PRC'], 'SVM')
    printEvaluationRes([rf_loss, rf_roc, rf_prc], ['LogLoss', 'AUC-ROC', 'AUC-PRC'], 'RF')
    # printEvaluationRes([bt_loss, bt_roc, bt_prc], ['LogLoss', 'AUC-ROC', 'AUC-PRC'], 'BT')
    
    # get the best model
    # svm = svm_mdl[svm_loss.index(min(svm_loss))]
    rf = rf_mdl[rf_loss.index(min(rf_loss))]
    # bt = bt_mdl[bt_loss.index(min(bt_loss))]
    
    # find the index of column which correspond to calss 1
    # idx = list(bt.classes_).index(1)
 
    # make predictions on knockdown, overexpression and all pert datasets
    for type in predictionData.keys():
        pert = pd.read_table(predictionData[type], header=0)
        # predict_and_save(svm, pert, 'SVM', type, disease)
        predict_and_save(rf, pert, 'RF', type, disease)
        # predict_and_save(bt, pert, 'GBM', type, disease)
