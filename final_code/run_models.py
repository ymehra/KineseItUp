## Authors: Yash Mehra, Gus Moir, Andrew Rose, Hans Schumann
## Version: June 2018
## 
## This file will be run from command line to create a user-specified model
## 

# import the necessary files
import getopt
import sys
import pandas as pd
import numpy as np
import json
import sklearn
import pickle
# from sklearn.neural_network import MLPClassifier
import script as sc
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score

# >>> py run___ user model_want data_file pickle_output_name 

# helper_functions
def save_pkl(classifier, filename):
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)
        print("Classifier saved as " + filename)


def svm(trainX, trainY, file):
    clf = sklearn.svm.SVC(cache_size=7000)
    clf.fit(trainX, trainY)
    save_pkl(clf, file)


def neural_net(trainX, trainY, file):
    n_net = sklearn.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2000, 1000),
                                                 random_state=1, verbose=True)
    n_net.fit(trainX, trainY)
    save_pkl(n_net, file)


def random_forest(trainX, trainY, file):
    rf = sklearn.ensemble.RandomForestClassifier(max_depth=5, random_state=0)
    rf.fit(trainX, trainY)
    save_pkl(rf, file)


def knn(trainX, trainY, file):
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=15)
    knn.fit(trainX, trainY)
    save_pkl(knn, file)


def gradient_boost(trainX, trainY, file):
    grad_boost = sklearn.ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=7)
    grad_boost.fit(trainX, trainY)
    save_pkl(grad_boost, file)


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
   
   if model == "gradient_boosting":
      raise NotImplementedError
   
   elif model == "svm":
      raise NotImplementedError
   
   if model == "knn":
      raise NotImplementedError
   
   if model == "rf":
      raise NotImplementedError
   
   if model == "neural_network":
      raise NotImplementedError

   
if __name__ == "__main__":
   main(sys.argv[1:])

