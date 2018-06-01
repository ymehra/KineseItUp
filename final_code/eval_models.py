# import the necessary files
import getopt
import sys
import pandas as pd
import numpy as np
import json
from sklearn.metrics import ConfusionMatrix
import pickle
import script as sc


def print_confusion_matrix(classifier, data):
    x_vars = ['mean.vm', 'sd.vm', 'mean.ang', 'sd.ang', 'p625', 'dfreq', 'ratio.df']
    predicted = classifier.predict(data)


def main(argv):
    input_file = "Not found"
    model = "Not found"
    output_file = "Not found"

    try:
        opts, args = getopt.getopt(argv, "hi:m:u:o:", ["ifile=", "mfile=", "ufile=", "ofile="])
    except getopt.GetoptError:
        print('run_models.py -i <inputfile> -m <model name> -u <user> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('eval_models.py -i <inputfile> -m <model name> -u <user> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-u", "--user"):
            user = arg
        elif opt in ("-o", "--output"):
            output_file = arg

    print('input file name given :', input_file)
    print('model given :', model)
    print('user given :', user)
    print('output file name given :', output_file)

    data = pd.read_csv(input_file)

    with open(model, 'rb') as pkl:
        classifier = pickle.load(pkl)

    print_confusion_matrix(classifier, data)


if __name__ == "__main__":
    main(sys.argv[1:])

