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
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import matplotlib.axes as ax
from scipy.stats import linregress

def print_confusion_matrix(data, domain):
    """
    This function will create the confusion matrices depending on which domain you want
    :param data: the csv file of the tested model
    :param domain: the activity assignment (domain) you are interested in
    :
    """
    domain = domain.lower()
    if domain == "overall":
        conf_mat = confusion_matrix(data['coding'],data['predicted'])
        a = conf_mat[0][0]
        b = conf_mat[0][1]
        c = conf_mat[1][0]
        d = conf_mat[1][1]
        print ("\tOverall Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Recall")
        print (str("Actual nonsed | " + str(a) + " | " + str(b) + " | " + str(round(a / (a+b),3)) ))
        print (str("        sed   | " + str(c) + " | " + str(d) + " | " + str(round(d / (c+d),3)) ))
        print (str("    Precision | " + str(round(a / (a+c),3)) + " | " + str(round(d / (b+d),3)) + " | " + str(round((a+d) / (a+b+c+d),3)) ))
        print ()
        print ("Overall Accuracy = ",sum(data['coding'] == data['predicted']) / len(data['coding']))
        print ()

    elif domain == "e":
        data_temp = data[data['type'].str.contains('E')]
        conf_mat = confusion_matrix(data_temp['coding'],data_temp['predicted'])
        a = conf_mat[0][0]
        b = conf_mat[0][1]
        c = conf_mat[1][0]
        d = conf_mat[1][1]
        print ("\tErrands Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Recall")
        print (str("Actual nonsed | " + str(a) + " | " + str(b) + " | " + str(round(a / (a+b),3)) ))
        print (str("        sed   | " + str(c) + " | " + str(d) + " | " + str(round(d / (c+d),3)) ))
        print (str("    Precision | " + str(round(a / (a+c),3)) + " | " + str(round(d / (b+d),3)) + " | " + str(round((a+d) / (a+b+c+d),3)) ))
        print ()
        print ("Errands Accuracy = ",sum(data_temp['coding'] == data_temp['predicted']) / len(data_temp['coding']))
        print ()

    elif domain == "a":
        data_temp = data[data['type'].str.contains('A')]
        conf_mat = confusion_matrix(data_temp['coding'],data_temp['predicted'])
        a = conf_mat[0][0]
        b = conf_mat[0][1]
        c = conf_mat[1][0]
        d = conf_mat[1][1]
        print ("\tActive Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Recall")
        print (str("Actual nonsed | " + str(a) + " | " + str(b) + " | " + str(round(a / (a+b),3)) ))
        print (str("        sed   | " + str(c) + " | " + str(d) + " | " + str(round(d / (c+d),3)) ))
        print (str("    Precision | " + str(round(a / (a+c),3)) + " | " + str(round(d / (b+d),3)) + " | " + str(round((a+d) / (a+b+c+d),3)) ))
        print ()
        print ("Active Accuracy = ",sum(data_temp['coding'] == data_temp['predicted']) / len(data_temp['coding']))
        print ()

    elif domain == "l":
        data_temp = data[data['type'].str.contains('L')]
        conf_mat = confusion_matrix(data_temp['coding'],data_temp['predicted'])
        a = conf_mat[0][0]
        b = conf_mat[0][1]
        c = conf_mat[1][0]
        d = conf_mat[1][1]
        print ("\tLeisure Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Recall")
        print (str("Actual nonsed | " + str(a) + " | " + str(b) + " | " + str(round(a / (a+b),3)) ))
        print (str("        sed   | " + str(c) + " | " + str(d) + " | " + str(round(d / (c+d),3)) ))
        print (str("    Precision | " + str(round(a / (a+c),3)) + " | " + str(round(d / (b+d),3)) + " | " + str(round((a+d) / (a+b+c+d),3)) ))
        print ()
        print ("Leisure Accuracy = ",sum(data_temp['coding'] == data_temp['predicted']) / len(data_temp['coding']))
        print ()

    elif domain == "w":
        data_temp = data[data['type'].str.contains('W')]
        conf_mat = confusion_matrix(data_temp['coding'],data_temp['predicted'])
        a = conf_mat[0][0]
        b = conf_mat[0][1]
        c = conf_mat[1][0]
        d = conf_mat[1][1]
        print ("\tWork Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Recall")
        print (str("Actual nonsed | " + str(a) + " | " + str(b) + " | " + str(round(a / (a+b),3)) ))
        print (str("        sed   | " + str(c) + " | " + str(d) + " | " + str(round(d / (c+d),3)) ))
        print (str("    Precision | " + str(round(a / (a+c),3)) + " | " + str(round(d / (b+d),3)) + " | " + str(round((a+d) / (a+b+c+d),3)) ))
        print ()
        print ("Work Accuracy = ",sum(data_temp['coding'] == data_temp['predicted']) / len(data_temp['coding']))
        print ()

    elif domain == "h":
        data_temp = data[data['type'].str.contains('H')]
        conf_mat = confusion_matrix(data_temp['coding'],data_temp['predicted'])
        a = conf_mat[0][0]
        b = conf_mat[0][1]
        c = conf_mat[1][0]
        d = conf_mat[1][1]
        print ("\tHousework Confusion Matrix")
        print ("\t         Predicted")
        print ("\t      | nonsed| sed   | Recall")
        print (str("Actual nonsed | " + str(a) + " | " + str(b) + " | " + str(round(a / (a+b),3)) ))
        print (str("        sed   | " + str(c) + " | " + str(d) + " | " + str(round(d / (c+d),3)) ))
        print (str("    Precision | " + str(round(a / (a+c),3)) + " | " + str(round(d / (b+d),3)) + " | " + str(round((a+d) / (a+b+c+d),3)) ))
        print ()
        print ("Housework Accuracy = ",sum(data_temp['coding'] == data_temp['predicted']) / len(data_temp['coding']))
        print ()

def create_plot(data,file,model):
    """
    This creates the regression plot of the actual and predicted values for each person
    :param data: the tested csv models
    :param file: the output file name
    :param model: used for naming the graph (the model you used)
    """
    model = model.replace("_"," ")
    ## creating the person data
    person_data = pd.DataFrame(columns = ['id','category','actual_sed','pred_sed','total'])
    ## iterate through the people
    for person in np.unique(data['type']):
        temp = data[data['type'] == person]
        ## add the actual and predicted sedentary classifications to the dataframe
        person_data.loc[len(person_data)] = [person,person[:1],np.sum(temp['coding'] == 'sedentary'),np.sum(temp['predicted'] == 'sedentary'),len(temp)]
    ## get the predicted and actual sedentary percentages from the dataframe values
    person_data['Pred_sed_pct'] = (person_data['pred_sed'] / person_data['total']).astype(float)
    person_data['Actual_sed_pct'] = (person_data['actual_sed'] / person_data['total']).astype(float)
    
    ## produce the linear regression and perfect fit line
    x = np.arange(2)
    y1 = x
    reg = linregress(person_data['Pred_sed_pct'],person_data['Actual_sed_pct'])
    y2 = reg[0]*x + reg[1]
    colors = np.array(['b','b','b','g','g','g','r','r','r','c','c','c','c','m','m','m'])

    ## create separate data frames for each of the domains
    a = person_data[person_data['category'] == 'A']
    e = person_data[person_data['category'] == 'E']
    l = person_data[person_data['category'] == 'L']
    w = person_data[person_data['category'] == 'W']
    h = person_data[person_data['category'] == 'H']

    ## change size
    plt.figure(figsize = (8,6))
    ## put in regression and perfect fit lines
    plt.plot(x,y1,'--',color = 'black')
    plt.plot(x,y2,'-',color = 'orange')
    ## put in the points for each domain
    plt.scatter(a['Pred_sed_pct'],a['Actual_sed_pct'],color = 'm')
    plt.scatter(e['Pred_sed_pct'],e['Actual_sed_pct'],color = 'c')
    plt.scatter(l['Pred_sed_pct'],l['Actual_sed_pct'],color = 'g')
    plt.scatter(w['Pred_sed_pct'],w['Actual_sed_pct'],color = 'b')
    plt.scatter(h['Pred_sed_pct'],h['Actual_sed_pct'],color = 'r')
    ## labeling stuff
    plt.xlabel('Predicted Sedentary Proportion')
    plt.ylabel('Actual Sedentary Proportion')
    plt.title(str(model + " Model Accuacy By Domain"),fontsize = 16)
    plt.legend(['Perfect Fit','Regression','Active','Errands','Leisure','Work','Housework'])
    plt.xlim([0,1])
    plt.ylim([0,1])
    ## put in the regression coefficient
    plt.text(x = 0.8, y = 0.1,s = ("r = " + str(round(reg[2],4))),fontsize = 16)
    ## save the plot in the desired png file
    plt.savefig(file)

def main(argv):
    input_file = "Not found"
    output_file = "Not found"

    try:
        opts, args = getopt.getopt(argv, "hi:o:m:", ["ifile=", "mfile=", "ofile="])
    except getopt.GetoptError:
        print('run_models.py -i <inputfile> -o <outputfile> -m <modelname>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('eval_models.py -i <inputfile> -m <model name> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--output"):
            output_file = arg
        elif opt in ("-m", "--kvalue"):
            model = arg


    print('input file name given :', input_file)
    print('output file name given :', output_file)
    print ('model given :', model)

    ## read in the results of the tested model
    data = pd.read_csv(input_file)

    ## print the confusion matrices for each domain
    print_confusion_matrix(data, "overall")
    print_confusion_matrix(data, "a")
    print_confusion_matrix(data, "e")
    print_confusion_matrix(data, "w")
    print_confusion_matrix(data, "l")
    print_confusion_matrix(data, "h")
    
    ## output graphs
    create_plot(data,output_file,model)

if __name__ == "__main__":
    main(sys.argv[1:])

