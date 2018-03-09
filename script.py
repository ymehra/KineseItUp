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
    data['start.time'] = pd.to_datetime(data['start.time'])
    observedData = data[(data['start.time'] >= timeStart) & (data['start.time'] <= timeEnd)]
    observedData['index'] = range(len(observedData))
    gt['index'] = gt['time']
    observedData = pd.merge(observedData,gt[['index','coding']])
    observedData.sort_index(inplace=True)
    copy = observedData[['mean.vm','sd.vm','mean.ang','sd.ang','p625','dfreq','ratio.df']].copy(deep=True)
    copy.loc[-1] = copy.loc[0]  # adding a row
    copy.index = copy.index + 1  # shifting index
    copy.sort_index(inplace=True)
    copy.columns = 'last.' + copy.columns
    observedData = pd.concat([observedData, copy], axis = 1)
    observedData = observedData.drop(observedData.index[len(observedData)-1])
    return observedData

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
        observedData1 = get_observed_data_for_subject(user, 1, struct[str(i)])
        observedData = pd.concat([observedData, observedData1])
        
    return observedData

