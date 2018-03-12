import pandas as pd
import json
import sklearn.svm as svm
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.neural_network import MLPClassifier

def load_Data(user, filename):
   data = json.load(open('dir.json'))
   return (data[user] + filename)

def useable_Data(data, gt, timeStart, timeEnd):
    data = get_lags(data, 3)
    data['start.time'] = pd.to_datetime(data['start.time'])
    observedData = data[(data['start.time'] >= timeStart) & (data['start.time'] <= timeEnd)]
    observedData['index'] = range(len(observedData))
    gt['index'] = gt['time']
    observedData = pd.merge(observedData,gt[['index','coding']])
    observedData.sort_index(inplace=True)
    return observedData

def get_lags(data, lags):
    prev_data = data[['mean.vm','sd.vm','mean.ang','sd.ang','p625','dfreq','ratio.df']]
    for i in range(lags):
        copy = prev_data.copy(deep=True)
        copy.iloc[-1,:] = copy.iloc[0,:]
        copy.index = copy.index + 1  # shifting index
        copy.sort_index(inplace=True)
        copy.columns = 'last.' + str(i+1) + "." + copy.columns
        prev_data = copy
        data = pd.concat([data, copy], axis = 1)
        data = data.drop(data.index[len(data)-1])
    return data

def select_features_from_lasso(X, y, alpha):
    # fit lasso model and pass to select from model
    lasso = LassoCV().fit(X, y)
    model = SelectFromModel(lasso, prefit=True)

    # new features
    X_new = model.transform(X)
    return X.columns[model.get_support()]

def load_struct():
    data = json.load(open('struct.json'))
    return data

def get_observed_data_for_subject(user, subject, files):
    data = pd.read_csv(load_Data(user, files["AG"]), header=0)
    gt1 = pd.read_csv(load_Data(user, files["GT1"]), header=0)
    gt2 = pd.read_csv(load_Data(user, files["GT2"]), header=0)
    type1 = files["GT1"][2]
    type2 = files["GT2"][2]
    start1 = files["StartTime1"]
    end1 = files["EndTime1"]
    start2 = files["StartTime2"]
    end2 = files["EndTime2"]
    observedData = pd.DataFrame(useable_Data(data, gt1, start1, end1))
    observedData['type'] = type1 + '-' + str(subject)
    observedData1 = pd.DataFrame(useable_Data(data, gt2, start2, end2))
    observedData1['type'] = type2 + '-' + str(subject)
    observedData = pd.concat([observedData, observedData1])
    return observedData

def get_all_subjects(user, files):
    observedData = pd.DataFrame()
    for i in files:
        observedData1 = get_observed_data_for_subject(user, i, files[str(i)])
        observedData = pd.concat([observedData, observedData1])    
    return observedData

def write_observedData(observedData, user):
    data = json.load(open('dir.json'))
    observedData.to_csv(data[user]+'complete.csv')

def get_complete(user):
     data = pd.read_csv(load_Data(user, 'complete.csv'))
     return data

def update_complete(user):
    files = load_struct()
    observedData = get_all_subjects(user, files)
    write_observedData(observedData, user)