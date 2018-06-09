## Authors: Yash Mehra, Gus Moir, Andrew Rose, Hans Schumann
## Version: June 2018
## 
## This file will be run from command line to create a user-specified model
## 

# preliminary things to get the script.py file
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# import the necessary files
import getopt
import pandas as pd
import numpy as np
import json
import sklearn
import sklearn.ensemble
import sklearn.neural_network
import sklearn.neighbors
import pickle
import math
import script as sc

# helper_functions
def save_pkl(classifier, filename):
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)
        print("Classifier saved as " + filename)

## functions for the various models -- these take in the training set
## and return the fitted model
def svm(trainX, trainY):
    clf = sklearn.svm.SVC(max_iter = 10)
    clf.fit(trainX, trainY)
    return clf

def neural_net(trainX, trainY):
    n_net = sklearn.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2000, 1000),
                                                 random_state=1, verbose=True)
    n_net.fit(trainX, trainY)
    return n_net

def random_forest(trainX, trainY):
   rf = sklearn.ensemble.RandomForestClassifier(max_depth=5, random_state=0)
   rf.fit(trainX, trainY)
   return rf

def knn(trainX, trainY):
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=15)
    knn.fit(trainX, trainY)
    return knn

def gradient_boost(trainX, trainY):
    grad_boost = sklearn.ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=7)
    grad_boost.fit(trainX, trainY)
    return grad_boost

def main(argv):
  input_file = "Not found"
  output_file = "Not found"
  user = "Not found"
  model = "Not found"
  cross_val = "all"

  try:
    opts, args = getopt.getopt(argv,"hi:m:u:o:c:",["ifile=","mfile=","ufile=", "ofile=", "cfile"])
  except getopt.GetoptError:
    print ('run_models.py -i <inputfile> -m <model name> -u <user> -o <outputfile> -c <crossvalmethod>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print ('run_models.py -i <inputfile> -m <model name> -u <user> -o <outputfile> -c <crossvalmethod>')
      sys.exit()
    elif opt in ("-i", "--ifile"):
      input_file = arg
    elif opt in ("-m", "--kvalue"):
      model = arg
    elif opt in ("-u", "--kvalue"):
      user = arg
    elif opt in ("-o", "--kvalue"):
      output_file = arg
    elif opt in ("-c", "--kvalue"):
      cross_val = arg

  print ('input file name given :', input_file)
  print ('model given :', model)
  print ('user given :', user)
  print ('output file name given :', output_file)
  print ('cross validation technique: ', cross_val)

  model = model.lower()
  cross_val = cross_val.lower()
   
  ## cross validation by person (leave-one-out)
  data = pd.DataFrame(sc.get_complete(user,input_file))
  data = sc.get_lags(data, 1)
  data = data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'start.time', 'index', 'activity'], axis=1)
  data_with_predictions = pd.DataFrame()

  ## define y variable
  y_var = 'coding'

  ## perform the cross validation (depending on user input -- thx Alex...)
  if cross_val == "default":
    pass

  elif cross_val == "leave_one_out":
    for person in np.unique(data['type']):
      ## split training and testing by one person
      train = data[data['type'] != person].copy(deep=True)
      test = data[data['type'] == person].copy(deep=True)

      ## subset x and y variables
      train_x = train.drop(['type', 'coding'], axis=1)
      train_y = train[y_var]
      test_x = test.drop(['type', 'coding'], axis=1)
      test_y = test[y_var]

      classifier = None

      if model == "gradient_boosting":
        classifier = gradient_boost(train_x,train_y)
      elif model == "svm":
        train_x = sklearn.preprocessing.scale(train_x)
        classifier = svm(train_x,train_y)
      elif model == "knn":
        classifier = knn(train_x,train_y)
      elif model == "rf":
        classifier = random_forest(train_x,train_y)
      elif model == "neural_network":
        classifier = neural_net(train_x,train_y)
      else:
        exit(1)

      ## use classifier to predict
      if model == "svm":
        test_x = sklearn.preprocessing.scale(test_x)
      pred = classifier.predict(test_x)
      test['predicted'] = pred

      ## print the accuracy of current person
      corr = test['predicted'] == test['coding']
      print (str(person + " accuracy = "),sum(corr) / len(corr))

      ## append the data set with predictions for that person
      data_with_predictions = data_with_predictions.append(test)

  ## k-fold cross validation
  elif cross_val.isdigit():
    people = np.unique(data['type'])
    np.random.shuffle(people)
    k = int(cross_val)
    how_many = (len(people) / k)
    
    for i in range(k):
      test_group = people[int(how_many * i):int(how_many * (i+1))]
      train = data[~data['type'].isin(test_group)].copy(deep=True)
      test = data[data['type'].isin(test_group)].copy(deep=True)

      ## subset x and y variables
      train_x = train.drop(['type', 'coding'], axis=1)
      train_y = train[y_var]
      test_x = test.drop(['type', 'coding'], axis=1)
      test_y = test[y_var]

      classifier = None

      if model == "gradient_boosting":
        classifier = gradient_boost(train_x,train_y)
      elif model == "svm":
        train_x = sklearn.preprocessing.scale(train_x)
        classifier = svm(train_x,train_y)
      elif model == "knn":
        classifier = knn(train_x,train_y)
      elif model == "rf":
        classifier = random_forest(train_x,train_y)
      elif model == "neural_network":
        classifier = neural_net(train_x,train_y)
      else:
        exit(1)

      ## use classifier to predict
      if model == "svm":
        test_x = sklearn.preprocessing.scale(test_x)
      pred = classifier.predict(test_x)
      test['predicted'] = pred

      ## print the accuracy of current person
      corr = test['predicted'] == test['coding']
      print ("People: ",test_group)
      print (str("Accuracy = "),sum(corr) / len(corr))
      print ()

      ## append the data set with predictions for that person
      data_with_predictions = data_with_predictions.append(test)

  elif cross_val == "domain":
    all_domains = ['A','W','E','L','H']
    for d in all_domains:
      ## get the ones from the domain
      subset = [v for i, v in enumerate(data['type']) if d in v]
      small_data = data[data['type'].isin(subset)]
      ## go through each person in the subgroup (domain)
      for person in np.unique(subset):
        ## split training and testing by one person
        train = small_data[small_data['type'] != person].copy(deep=True)
        test = small_data[small_data['type'] == person].copy(deep=True)

        ## subset x and y variables
        train_x = train.drop(['type', 'coding'], axis=1)
        train_y = train[y_var]
        test_x = test.drop(['type', 'coding'], axis=1)
        test_y = test[y_var]

        classifier = None

        if model == "gradient_boosting":
          classifier = gradient_boost(train_x,train_y)
        elif model == "svm":
          train_x = sklearn.preprocessing.scale(train_x)
          classifier = svm(train_x,train_y)
        elif model == "knn":
          classifier = knn(train_x,train_y)
        elif model == "rf":
          classifier = random_forest(train_x,train_y)
        elif model == "neural_network":
          classifier = neural_net(train_x,train_y)
        else:
          exit(1)

        ## use classifier to predict
        if model == "svm":
          test_x = sklearn.preprocessing.scale(test_x)
        pred = classifier.predict(test_x)
        test['predicted'] = pred

        ## print the accuracy of current person
        corr = test['predicted'] == test['coding']
        print (str(person + " accuracy = "),sum(corr) / len(corr))

        ## append the data set with predictions for that person
        data_with_predictions = data_with_predictions.append(test)


  ## train classifier on all the data
  if model == "gradient_boosting":
    classifier = gradient_boost(data.drop(['coding', 'type'], axis=1), data[y_var])
  elif model == "svm":
    classifier = svm(data.drop(['coding', 'type'], axis=1), data[y_var])
  elif model == "knn":
    classifier = knn(data.drop(['coding', 'type'], axis=1), data[y_var])
  elif model == "rf":
    classifier = random_forest(data.drop(['coding', 'type'], axis=1), data[y_var])
  elif model == "neural_network":
    classifier = neural_net(data.drop(['coding', 'type'], axis=1), data[y_var])
  else:
    exit(1)

  ## output the classifer and data_set with predictions
  save_pkl(classifier, output_file)

  ## write the data to a csv
  data_with_predictions.to_csv(str(model+'_new_')+input_file,index = False)
  ## print the cross-val score
  correct = data_with_predictions['coding'] == data_with_predictions['predicted']
  print ("Overall Model Accuracy = ",sum(correct) / len(correct))

if __name__ == "__main__":
  main(sys.argv[1:])

