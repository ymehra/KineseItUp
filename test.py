import pandas as pd
import json
from tabulate import tabulate

def load_Data(user, filename):
   data = json.load(open('dir.json'))
   dir = data[user] + filename
   data = pd.read_csv(dir)
   return data

def main():
   data = load_Data('Yash', 'AG01-02.csv')
   print tabulate(data.head(), headers='keys', tablefmt='psql')


main()