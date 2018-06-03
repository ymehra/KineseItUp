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
    print("clf made")
    clf.fit(trainX, trainY)
    print("fit complete")
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

   try:
      opts, args = getopt.getopt(argv,"hi:m:u:o:",["ifile=","mfile=","ufile=", "ofile="])
   except getopt.GetoptError:
      print ('run_models.py -i <inputfile> -m <model name> -u <user> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('run_models.py -i <inputfile> -m <model name> -u <user> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         input_file = arg
      elif opt in ("-m", "--kvalue"):
         model = arg
      elif opt in ("-u", "--kvalue"):
         user = arg
      elif opt in ("-o", "--kvalue"):
         output_file = arg

   print ('input file name given :', input_file)
   print ('model given :', model)
   print ('user given :', user)
   print ('output file name given :', output_file)

   model = model.lower()
   
   ## cross validation by person (leave-one-out)
   data = pd.DataFrame(sc.get_complete(user,input_file))
   data = sc.get_lags(data, 1)
   data = data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'start.time', 'index', 'activity'], axis=1)
   print(data.head())
   data_with_predictions = pd.DataFrame()

   ## define variables
   #x_vars = ['mean.vm','sd.vm','mean.ang','sd.ang','p625','dfreq','ratio.df']
   y_var = 'coding'

   for person in np.unique(data['type']):
      print (person)
      ## split training and testing by one person
      train = data[data['type'] != person].copy(deep=True)
      test = data[data['type'] == person].copy(deep=True)

      #print(train.head)
      
      ## subset x and y variables
      train_x = train.drop(['type', 'coding'], axis=1)
      train_y = train[y_var]
      test_x = test.drop(['type', 'coding'], axis=1)
      test_y = test[y_var]

      print(train_x.head())

      classifier = None
      print ("here")

      if model == "gradient_boosting":
         classifier = gradient_boost(train_x,train_y)
      elif model == "svm":
         print ("inside")
         classifier = svm(train_x,train_y)
         print("Finished SVM class")
      elif model == "knn":
         classifier = knn(train_x,train_y)
      elif model == "rf":
         classifier = random_forest(train_x,train_y)
      elif model == "neural_network":
         classifier = neural_net(train_x,train_y)
      else:
         print ("why?")
         exit(1)
      print("out of condtional")
      ## use classifier to predict
      pred = classifier.predict(test_x)
      test['predicted'] = pred
      print ("outside")
      ## print the accuracy of current person
      corr = test['predicted'] == test['coding']
      print (str(person + " accuracy = "),sum(corr) / len(corr))

      ## append the data set with predictions for that person
      data_with_predictions = data_with_predictions.append(test)
      print ("bottom")

   ## train classifier on all the data
   if model == "gradient_boosting":
      classifier = gradient_boost(data.drop(['coding', 'type'], axis=1), data[y_var])
   elif model == "svm":
      print("inside")
      classifier = svm(data.drop(['coding', 'type'], axis=1), data[y_var])
      print("Finished SVM class")
   elif model == "knn":
      classifier = knn(data.drop(['coding', 'type'], axis=1), data[y_var])
   elif model == "rf":
      classifier = random_forest(data.drop(['coding', 'type'], axis=1), data[y_var])
   elif model == "neural_network":
      classifier = neural_net(data.drop(['coding', 'type'], axis=1), data[y_var])
   else:
      print("why?")
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

