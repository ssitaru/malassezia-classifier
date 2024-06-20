from fronni import classification 
import pandas as pd

df = pd.read_csv('out.csv')

print(classification.classification_report(label = df.loc[:,'real'], predicted = df.loc[:,'predicted']))