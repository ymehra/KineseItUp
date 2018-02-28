import pandas as pd
import json

def load_Data(user, filename):
   data = json.load(open('dir.json'))
   dir = data[user] + filename
   data = pd.read_csv(dir)
   return data

def main():
   data = load_Data('Yash', '')
   data.head()

main()