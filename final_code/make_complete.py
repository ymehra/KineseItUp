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
import pickle
import script as sc


def main(argv):
    output_file = "Not found"
    user = "Not found"

    try:
        opts, args = getopt.getopt(argv, "hi:m:u:o:", ["ifile=", "mfile=", "ufile=", "ofile="])
    except getopt.GetoptError:
        print('make_complete.py -u <user> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('make_complete.py -u <user> -o <outputfile>')
            sys.exit()
        elif opt in ("-u", "--user"):
            user = arg
        elif opt in ("-o", "--outputfile"):
            output_file = arg

    print("Building complete dataset", output_file, "for user", user)
    sc.update_complete(user, output_file)
    print("Finished")


if __name__ == '__main__':
    main(sys.argv[1:])