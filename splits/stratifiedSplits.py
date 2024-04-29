import os
import mygene
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

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
    
def getSplits(X_orig, y_orig):
    trainFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    return [(list(train_index), list(test_index)) for train_index, test_index in trainFold.split(X_orig, y_orig)]
    
def getGeneName(geneId):
    mg = mygene.MyGeneInfo()
    return [g['symbol'] for g in mg.getgenes(geneId)]

if __name__ == '__main__':
    # ind = 'antipsychotics'
    # indication_dict =  {'antidepression':'/exeh_3/kai/data/ATC_antidepression.csv',
    #                     'antipsychotics':'/exeh_3/kai/data/ATC_antipsychotics.csv',
    #                     'anxiety_depression':'/exeh_3/kai/data/MEDI_anxiety_depression.csv',
    #                     'scz':'/exeh_3/kai/data/MEDI_scz.csv'}

    os.chdir("/exeh_3/kai/GE_result/genePerturbation")
    pheno = pd.read_csv('cmap_drug_expression_profile.csv', header=0)
    
    # generate indication
    # sePattern = getSearchPattern(indication_dict[ind])
    # indication = getIndication(sePattern, pheno['drugName'])
    
    X_orig = pheno.iloc[:, 2:].values
    # y_orig = np.asarray(indication)
    
