from __future__ import print_function

import warnings
import numpy as np
import pandas as pd
import os
import re
# import sys
import h5py
# from functools import partial

from sklearn.utils import shuffle, class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from keras.models import Sequential, load_model
from keras.regularizers import l1l2
from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils # For y values
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta, Adam, RMSprop, Adagrad, Adamax
from keras.regularizers import l1,l2, l1l2, activity_l2, activity_l1, activity_l1l2

# deactive warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings("ignore")
# warnings.filterwarnings('ignore', message='Changing the shape of non-C contiguous array')  # filter out specific warning

# Key: for unbalanced dataset, be careful with the choice of minibatch size in each epoch, if you choose too few, eg 50, then all validation sample in the minibatch may have the same label (validation loss will be the the same if all outcome variable is 0) 

def define_params_gid():

    return {
        'units1': [1000, 1500],
        'units2': [500, 1000],
        # 'units1': [1500, 2000],
        # 'units2': [1000, 1500],
        
        # 'dropout': [0.2, 0.4, 0.6, 0.8],
        'dropout': [0.6, 0.7, 0.8],  # for antipsycho
        # 'dropout': [0.6, 0.7, 0.8],  # for scz

        # 'batch_size' : hp.uniform('batch_size', 256,1024),
        # 'batch_size' : [input_num], # key change for unbalanced dataset 
        
        'nb_epoch': [10, 20],
        'l1': np.logspace(-13, -11, 3),
        'l2': np.logspace(-9, -8, 2)
        # 'optimizer': ['adadelta','adam','rmsprop'],
        # 'activation': ['relu','softplus','tanh'] 
    }

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

def write_file(file_name, drug_name, predict_result, real_result):
    assert len(drug_name) == len(predict_result) and len(predict_result) == len(real_result), "The length of arrays is (%s, %s, %s) not the same!" % (len(drug_name), len(predict_result), len(predict_result))

    with open(file_name, 'a') as f:
        for i in range(len(drug_name)):
            f.write('%s, %s, %s\n' % (drug_name[i], predict_result[i], real_result[i])) 
                                                            
def construct_model(params):
 
    model = Sequential()
    model.add(Dense(output_dim=params['units1'],
                    W_regularizer=l1l2(l1=params['l1'], l2=params['l2']),
                    input_dim = 7467)) 
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(output_dim=params['units2'],
                    W_regularizer=l1l2(l1=params['l1'], l2=params['l2'])))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(params['dropout']))
    
    # if params['choice']['layers'] == 'three':
    #     model.add(Dense(output_dim=params['choice']['units3'], W_regularizer=l1l2(l1=params['l1_1st_layer'],l2=params['l2_1st_layer']), init = "glorot_uniform")) 
    #     model.add(BatchNormalization())
    #     model.add(Activation(params['activation']))
    #     model.add(Dropout(params['choice']['dropout3']))    
    
    model.add(Dense(1))
    # model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model
    
def create_fit(X_inner_train, y_inner_train, params, checkfile, class_weights = None, val_data = None, is_print=0, type='train'):
    h5py.File(checkfile,'w').close()
    model = construct_model(params)
    callback1 = ModelCheckpoint(filepath = checkfile, verbose = 0, save_weights_only = False, save_best_only = True)
    model.fit(X_inner_train, y_inner_train, 
              batch_size = y_inner_train.shape[0],
              show_accuracy = True,
              nb_epoch = params['nb_epoch'], 
              validation_data = val_data, 
              class_weight = class_weights,
              callbacks = [callback1],
              verbose = is_print)
    
    # inefficient method to rebuild model from check file
    # model_json = model.to_json()
    # del(model)
    # model = model_from_json(model_json)
    # model.load_weights(checkfile)
    if type == 'evaluate':
        return model
    else:
        model = load_model(checkfile) 
        model.compile(loss='binary_crossentropy', optimizer='adadelta')
        return model

def gridSearchCV(X_train, y_train, grid_params, sample_weights, n_fold = 3, class_weights = None):    
    global train_score, val_score, roc_score, prc_score
    loop_num = -1
    length = len(grid_params)
    train_score, val_score = np.zeros((length, n_fold), dtype=float), np.zeros((length, n_fold), dtype=float)
    loss_score, roc_score, prc_score = np.zeros((length, n_fold), dtype=float), np.zeros((length, n_fold), dtype=float), np.zeros((length, n_fold), dtype=float)

    trainFold = StratifiedKFold(n_splits = n_fold)
    for train_idx, test_idx in trainFold.split(X_train, y_train):
        loop_num += 1
        X_inner_train, X_inner_test = X_train[train_idx], X_train[test_idx]
        y_inner_train, y_inner_test = y_train[train_idx], y_train[test_idx]
        train_weights, test_weights = sample_weights[train_idx], sample_weights[test_idx]
        validation_data = (X_inner_test, y_inner_test)
        
        for i in range(length):
            
            model = create_fit(X_inner_train, y_inner_train, grid_params[i], train_checkfile, class_weights = class_weights, val_data = validation_data)
            
            training_loss = model.evaluate(X_inner_train, y_inner_train, batch_size = y_inner_train.shape[0], verbose = 0, sample_weight = train_weights)
            train_score[i][loop_num] = training_loss

            validation_loss = model.evaluate(X_inner_test, y_inner_test, batch_size = y_inner_test.shape[0], verbose = 0, sample_weight = test_weights)
            val_score[i][loop_num] = validation_loss
            
            # compute AUC of ROC and PRC
            pred = model.predict(X_inner_test, batch_size = X_inner_test.shape[0], verbose = 0)
            prc_score[i][loop_num] = average_precision_score(y_inner_test, pred[:,0]) 
            roc_score[i][loop_num] = roc_auc_score(y_inner_test, pred[:,0]) 
            loss_score[i][loop_num] = log_loss(y_inner_test, pred[:,0]) 
            
            print('%0.3f/%0.3f for %r' % (training_loss, validation_loss, grid_params[i]))
            del(model)
    
    mean_val_loss = np.average(val_score, axis = 1)
    idx = np.argmin(mean_val_loss)
    best_param = grid_params[idx]
    print('Best parameters: ', best_param, '; corresponding training and validation loss: ', train_score[idx], " ,", mean_val_loss[idx])
    print('LogLoss for the best parameters:', loss_score[idx,:])
    print('AUC-ROC for the best parameters:', roc_score[idx,:])
    print('PRC-ROC for the best parameters:', prc_score[idx,:])

    best_model = create_fit(X_train, y_train, best_param, eval_checkfile, class_weights = class_weights, type='evaluate')
    
    return {'train_score': train_score, 'validation_score': val_score, 'best_model': best_model}


def run(X_orig, y_orig, indication, weights = None, n_fold = 3, is_print = False):
    global X_test, y_test, train_checkfile, eval_checkfile
    
    neg_log_loss = []
    # file_name = 'DNN_weighted_%s_prediction.out' % indication
    train_checkfile, eval_checkfile = "callback_train_%s_weighted.h5" % indication, "callback_test_%s_weighted.h5" % indication
    # os.system('touch -f %s && > %s' % (file_name, file_name))
    
    # compute balanced class weight
    class_weights = class_weight.compute_class_weight(classes = np.unique(y_orig), y = y_orig, class_weight = 'balanced')
    weights = dict(enumerate(class_weights))
    sample_weights = np.repeat(class_weights[0], y_orig.shape[0])
    sample_weights[y_orig == 1] = class_weights[1]
    print('class weights: ', weights)
    
    # shuffle before training
    X_orig, y_orig, sample_weights = shuffle(X_orig, y_orig, sample_weights, random_state=0)

    # create parameter grid
    params = define_params_gid()
    grid_params = list(ParameterGrid(params))

    res = gridSearchCV(X_orig, y_orig, grid_params, sample_weights, class_weights= weights)
    # mean_train_score = np.average(res['train_score'], axis = 1)
    # mean_val_score = np.average(res['validation_score'], axis = 1)
    
    # show train loss, validation loss and corresponding params
    # if is_print == True:
    #     for train, val, param in zip(mean_train_score, mean_val_score, grid_params):
    #         print('%0.3f/%0.3f for %r' % (train, val, param))
        
    #     print('Best parameters: ', grid_params[idx], ', and corresponding validation loss: ', np.argmin(mean_val_score))
    
    # test_loss = best_model.evaluate(X_test, y_test, batch_size = y_test.shape[0], verbose = 0)
    # print('prediction loss on the test set: %s\n' % test_loss)
    
    # pred = best_model.predict(X_test, batch_size = X_test.shape[0], verbose = 0) 
    # write_file(file_name, drug_name_test, pred[:,0], y_test)
    
    # neg_log_loss.append(test_loss)
    return res['best_model']

def obtainDataset(indication_loc):
    pheno = pd.read_csv('cmap_drug_expression_profile.csv', header=0)
    
    # generate indication
    sePattern = getSearchPattern(indication_loc)
    indication = getIndication(sePattern, pheno['drugName'])
    
    X_orig = pheno.iloc[:, 2:].values
    y_orig = np.asarray(indication)
    return(X_orig, y_orig)

def predict_and_save(fitted_obj, pert, estimator_name, pertType, ind):
    X = pert.iloc[:, 1:].values
    pertRes = fitted_obj.predict(X, batch_size = X.shape[0], verbose = 0)
    # print('The shape of prediction: ', pertRes.shape)
    resDict = {'pertId': pert.loc[:, 'perturbagen'], "predRet": pd.Series(pertRes.reshape((pertRes.shape[0],)))}
    pred_res = pd.DataFrame(resDict)
    pred_res.to_csv(estimator_name + '_' + pertType + '_' + ind + '_Res.csv', index = False)

if __name__ == '__main__':
    # num = sys.argv[1]
    # data_source = sys.argv[2]
    os.chdir("/exeh_3/kai/GE_result/genePerturbation")
    
    indication_dict =  {'antidepression':'/exeh_3/kai/data/ATC_antidepression.csv',
                        'antipsychotics':'/exeh_3/kai/data/ATC_antipsychotics.csv',
                        'anxiety_depression':'/exeh_3/kai/data/MEDI_anxiety_depression.csv',
                        'scz':'/exeh_3/kai/data/MEDI_scz.csv'}    
 
    predictionData = {'knockdown': 'consensi-knockdown.tsv',
                      'overexpression':'consensi-overexpression.tsv',
                      'pert_id':'consensi-pert_id.tsv'}
    
    ind = 'antipsychotics'
    
    X_orig, y_orig = obtainDataset(indication_dict[ind])
    
    # run the repeated cross-validation
    optimal_model = run(X_orig, y_orig, ind)
    
    for tp in predictionData.keys():
        pert = pd.read_table(predictionData[tp], header = 0)
        predict_and_save(optimal_model, pert, 'Deep_learning', tp, ind)
    
    # tranformed_res = zip(*model_result)
    # print("The negative log loss for deep neuron networks: %s, average log loss: %s" % (model_result, np.mean(model_result)))
