#/usr/bin/env python

# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, average_precision_score, log_loss, roc_auc_score
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import StratifiedKFold

from ClassificationModels import SVM, RandomForest, GradientBoostingMachine

def showGridScore(fitObj, metrics):
    ''' print mean_test_score, std_test_score and corresponding parameters 
        Args:
            fitObj: the fitted GridSearchCV/RandomSearchCV object
    '''
    train_means = fitObj.cv_results_['mean_train_%s' % metrics]
    test_means = fitObj.cv_results_['mean_test_%s' % metrics]
    params = fitObj.cv_results_['params']
    
    for train, test, param in zip(train_means, test_means, params):
        print('%0.3f / %0.3f for %r' % (train, test, param))
    print('Best parameters and score: ', fitObj.best_params_, fitObj.best_score_, '\n')

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

def computeSampleWeights(y, is_print = True):
    class_weights = class_weight.compute_class_weight(classes = np.unique(y), y = y, class_weight = 'balanced')
    weights = np.repeat(class_weights[0], y.shape[0])
    weights[y == 1] = class_weights[1]
    
    if is_print == True:
        print("Class weights: ", class_weights)
    return weights

def roc_prc(y_true, y_pred):
    '''compute ACU for ROC and PRC
    y_true: array of true label of observations, values should be either 1 or 0
    y_pred: array of predicted prob for all observations, values are within [0, 1]
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label = 1)  
    roc_auc = auc(fpr, tpr)
    prc_auc = average_precision_score(y_true, y_pred, average="micro")
    return(roc_auc, prc_auc)

def printInfo(fittedObj, scoring):
    idx = fittedObj.best_index_
    for s in scoring.keys():
        print(s, ":", fittedObj.cv_results_["split0_test_%s" % s][idx], \
              fittedObj.cv_results_["split1_test_%s" % s][idx], \
              fittedObj.cv_results_["split2_test_%s" % s][idx])

def computeMetrics(y_true, y_pred, mdl):
    logloss = log_loss(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    prc = average_precision_score(y_true, y_pred)
    print('Metrics for ', mdl,':', logloss, roc, prc)
    
def predict_and_save(fitted_obj, pert, estimator_name, pertType, ind):
    pertRes = fitted_obj.predict_proba(pert.iloc[:, 1:].values)
    resDict = {"predRes": pd.Series(pertRes[:,1]), 'pertId': pert.loc[:, 'perturbagen']}
    pred_res = pd.DataFrame(resDict)
    pred_res.to_csv(estimator_name + '_' + pertType + '_' + ind + '_Res.csv', index = False)
    
def main():
    os.chdir('/exeh_3/kai/GE_result/genePerturbation')
    indication_dict =  {'antidepression':'/exeh_3/kai/data/ATC_antidepression.csv',
                        'antipsychotics':'/exeh_3/kai/data/ATC_antipsychotics.csv',
                        'anxiety_depression':'/exeh_3/kai/data/MEDI_anxiety_depression.csv',
                        'scz':'/exeh_3/kai/data/MEDI_scz.csv'}
    
    ind = 'antipsychotics'
    pheno = pd.read_csv('cmap_drug_expression_profile.csv', header = 0)
    
    # generate indication
    sePattern = getSearchPattern(indication_dict[ind])
    # sePattern = getSearchPattern('/exeh_3/kai/data/N06A.txt') # antidepressant
    indication = getIndication(sePattern, pheno['drugName'])
    
    # .values would transfer dataframe to an array
    # .iloc is a function to index row/columns in dataframe
    X_orig = pheno.iloc[:, 2:].values
    y_orig = np.asarray(indication)
    
    # compute sample_weights
    weights = computeSampleWeights(y_orig)
    
    X_orig, y_orig, weights = shuffle(X_orig, y_orig, weights, random_state = 0)
    trainFold = StratifiedKFold(n_splits = 3)

    # create, fit model and yield cross validation result
    # params = getParams()
    # trainFold = StratifiedKFold(n_splits = 3)
    scoring = {"LogLoss": "neg_log_loss", "PRC": make_scorer(average_precision_score), "AUC": "roc_auc"}
    svm_params = {'C': [5, 10, 50, 100], 'gamma': np.logspace(-8, -3, 6)} 
    SVM_obj = SVM(svm_params, num_jobs = 6, cv = trainFold, metrics = scoring, refit = 'LogLoss')
    SVM_obj.fit(X_orig, y_orig, sample_weight = weights)
    
    RF_obj = RandomForest(num_jobs = 6, cv = trainFold, metrics = scoring, refit = 'LogLoss')
    RF_obj.fit(X_orig, y_orig, sample_weight = weights)
    
    GBM_obj = GradientBoostingMachine(num_jobs = 6, cv = trainFold, metrics = scoring, refit = 'LogLoss')
    GBM_obj.fit(X_orig, y_orig, sample_weight = weights) 
   
    # print grid search 
    showGridScore(SVM_obj, 'LogLoss')
    showGridScore(RF_obj, 'LogLoss')
    showGridScore(GBM_obj, 'LogLoss')
 
    # print LogLoss, AUC and PRC for the best parameter chosen by LogLoss
    # printInfo(SVM_obj, scoring)
    # printInfo(RF_obj, scoring)
    # printInfo(GBM_obj, scoring)
    
    for train_index, test_index in trainFold.split(X_orig, y_orig):
        X_train, y_train = X_orig[train_index], y_orig[train_index]
        X_test, y_test = X_orig[test_index], y_orig[test_index]
      
        # calculate metrics
        y_pred = SVM_obj.predict_proba(X_test)[:, 1]
        computeMetrics(y_test, y_pred, 'SVM')
        y_pred = GBM_obj.predict_proba(X_test)[:, 1]
        computeMetrics(y_test, y_pred, 'GBM')
        y_pred = RF_obj.predict_proba(X_test)[:, 1]
        computeMetrics(y_test, y_pred, 'RF')
    
    predictionData = {'knockdown': 'consensi-knockdown.tsv',
                      'overexpression':'consensi-overexpression.tsv',
                      'pert_id':'consensi-pert_id.tsv'}
    
    # make predictions on knockdonw, overexpression and all pert datasets
    for type in predictionData.keys():
        pert = pd.read_table(predictionData[type], header=0)
        predict_and_save(SVM_obj, pert, 'SVM', type, ind)
        predict_and_save(RF_obj, pert, 'RF', type, ind)
        predict_and_save(GBM_obj, pert, 'GBM', type, ind)

if __name__ == '__main__':
    main()
