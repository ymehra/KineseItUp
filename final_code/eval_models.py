## Authors: Yash Mehra, Gus Moir, Andrew Rose, Hans Schumann
## Version: June 2018
## 
## This file will be run from command line to evaluate the generated models
## 

# preliminary things to get the script.py file
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# import the necessary files
import getopt
import sys
import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import pickle
import script as sc


# def print_confusion_matrix(classifier, data):
#     # x_vars = ['mean.vm', 'sd.vm', 'mean.ang', 'sd.ang', 'p625', 'dfreq', 'ratio.df']
#     # predicted = classifier.predict(data)
#     print (confusion_matrix(data['coding'],data['predicted']))

def print_confusion_matrix(data, domain):
    domain = domain.lower()
    if domain == "overall":
        conf_mat = confusion_matrix(data['coding'],data['predicted'])
        print ("\tOverall Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Total")
        print (str("Actual nonsed | " + str(conf_mat[0][0]) + " | " + str(conf_mat[0][1]) + " | " + str(conf_mat[0][0] + conf_mat[0][1]) ))
        print (str("        sed   | " + str(conf_mat[1][0]) + " | " + str(conf_mat[1][1]) + " | " + str(conf_mat[1][0] + conf_mat[1][1]) ))
        print (str("        Total | " + str(conf_mat[0][0] + conf_mat[1][0]) + " | " + str(conf_mat[0][1] + conf_mat[1][1]) + " | " + str(len(data)) ))
        print ()
        print ("Overall Accuracy = ",sum(data['coding'] == data['predicted']) / len(data['coding']))
        print ()

    elif domain == "e":
        data_temp = data[data['type'].str.contains('E')]
        conf_mat = confusion_matrix(data_temp['coding'],data_temp['predicted'])
        print ("\tErrands Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Total")
        print (str("Actual nonsed | " + str(conf_mat[0][0]) + " | " + str(conf_mat[0][1]) + " | " + str(conf_mat[0][0] + conf_mat[0][1]) ))
        print (str("        sed   | " + str(conf_mat[1][0]) + " | " + str(conf_mat[1][1]) + " | " + str(conf_mat[1][0] + conf_mat[1][1]) ))
        print (str("        Total | " + str(conf_mat[0][0] + conf_mat[1][0]) + " | " + str(conf_mat[0][1] + conf_mat[1][1]) + " | " + str(len(data_temp)) ))
        print ()
        print ("Errands Accuracy = ",sum(data_temp['coding'] == data_temp['predicted']) / len(data_temp['coding']))
        print ()

    elif domain == "a":
        data_temp = data[data['type'].str.contains('A')]
        conf_mat = confusion_matrix(data_temp['coding'],data_temp['predicted'])
        print ("\tActive Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Total")
        print (str("Actual nonsed | " + str(conf_mat[0][0]) + " | " + str(conf_mat[0][1]) + " | " + str(conf_mat[0][0] + conf_mat[0][1]) ))
        print (str("        sed   | " + str(conf_mat[1][0]) + " | " + str(conf_mat[1][1]) + " | " + str(conf_mat[1][0] + conf_mat[1][1]) ))
        print (str("        Total | " + str(conf_mat[0][0] + conf_mat[1][0]) + " | " + str(conf_mat[0][1] + conf_mat[1][1]) + " | " + str(len(data_temp)) ))
        print ()
        print ("Active Accuracy = ",sum(data_temp['coding'] == data_temp['predicted']) / len(data_temp['coding']))
        print ()

    elif domain == "l":
        data_temp = data[data['type'].str.contains('L')]
        conf_mat = confusion_matrix(data_temp['coding'],data_temp['predicted'])
        print ("\tLeisure Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Total")
        print (str("Actual nonsed | " + str(conf_mat[0][0]) + " | " + str(conf_mat[0][1]) + " | " + str(conf_mat[0][0] + conf_mat[0][1]) ))
        print (str("        sed   | " + str(conf_mat[1][0]) + " | " + str(conf_mat[1][1]) + " | " + str(conf_mat[1][0] + conf_mat[1][1]) ))
        print (str("        Total | " + str(conf_mat[0][0] + conf_mat[1][0]) + " | " + str(conf_mat[0][1] + conf_mat[1][1]) + " | " + str(len(data_temp)) ))
        print ()
        print ("Leisure Accuracy = ",sum(data_temp['coding'] == data_temp['predicted']) / len(data_temp['coding']))
        print ()

    elif domain == "w":
        data_temp = data[data['type'].str.contains('W')]
        conf_mat = confusion_matrix(data_temp['coding'],data_temp['predicted'])
        print ("\tWork Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Total")
        print (str("Actual nonsed | " + str(conf_mat[0][0]) + " | " + str(conf_mat[0][1]) + " | " + str(conf_mat[0][0] + conf_mat[0][1]) ))
        print (str("        sed   | " + str(conf_mat[1][0]) + " | " + str(conf_mat[1][1]) + " | " + str(conf_mat[1][0] + conf_mat[1][1]) ))
        print (str("        Total | " + str(conf_mat[0][0] + conf_mat[1][0]) + " | " + str(conf_mat[0][1] + conf_mat[1][1]) + " | " + str(len(data_temp)) ))
        print ()
        print ("Work Accuracy = ",sum(data_temp['coding'] == data_temp['predicted']) / len(data_temp['coding']))
        print ()

    elif domain == "h":
        data_temp = data[data['type'].str.contains('H')]
        conf_mat = confusion_matrix(data_temp['coding'],data_temp['predicted'])
        print ("\tHousework Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Total")
        print (str("Actual nonsed | " + str(conf_mat[0][0]) + " | " + str(conf_mat[0][1]) + " | " + str(conf_mat[0][0] + conf_mat[0][1]) ))
        print (str("        sed   | " + str(conf_mat[1][0]) + " | " + str(conf_mat[1][1]) + " | " + str(conf_mat[1][0] + conf_mat[1][1]) ))
        print (str("        Total | " + str(conf_mat[0][0] + conf_mat[1][0]) + " | " + str(conf_mat[0][1] + conf_mat[1][1]) + " | " + str(len(data_temp)) ))
        print ()
        print ("Housework Accuracy = ",sum(data_temp['coding'] == data_temp['predicted']) / len(data_temp['coding']))
        print ()

        

def main(argv):
    input_file = "Not found"
    output_file = "Not found"

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "mfile=", "ufile=", "ofile="])
    except getopt.GetoptError:
        print('run_models.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('eval_models.py -i <inputfile> -m <model name> -u <user> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--output"):
            output_file = arg

    print('input file name given :', input_file)
    print('output file name given :', output_file)

    data = pd.read_csv(input_file)

    # with open(model, 'rb') as pkl:
    #     classifier = pickle.load(pkl)

    # print_confusion_matrix(classifier, data)
    print_confusion_matrix(data, "overall")
    print_confusion_matrix(data, "a")
    print_confusion_matrix(data, "e")
    print_confusion_matrix(data, "w")
    print_confusion_matrix(data, "l")
    print_confusion_matrix(data, "h")
    
    ## output to text file
    f = open(output_file,'w')
    f.write(print_confusion_matrix(data,"overall"))


if __name__ == "__main__":
    main(sys.argv[1:])

