import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.linear_model import logisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def simple_graphing(data):
	positive = data.loc[data['Class'] == 1].values
	negative = data.loc[data['Class'] == 0].values
	bar_data = [len(positive), len(negative)]
	plt.bar(height = bar_data, left = [0,1])
	plt.show()
	
def new_sample(data):
	#Sample size = 2*N
	N = len(data.loc[data['Class'] == 1].values)
	n_negative = len(data.loc[data['Class'] == 0].values)
	sample_index = random.sample(range(0, n_negative), N) 
	cols = list(data.columns.values)
	
	data_sample = data.iloc[sample_index]
	data_sample = pd.concat([data_sample, data.loc[data['Class'] == 1]])
	return data_sample

if __name__ == '__main__':
	file = 'creditcard.csv'
	data = pd.read_csv(file, delimiter=',')
	
	#Uncomment to see simple graphs of the given data
	#simple_graphing(data)		
	data_sample = new_sample(data)
	
	#Separate input data
	Y = data_sample['Class'].values
	data_sample = data_sample.drop('Class', axis = 1)
	X = data_sample.values
	
	#Scale input data
	X = preprocessing.scale(X)
	#Select val size
	val_size = 0.1
	
	#Split Train and validation sets
	X_train, X_val, y_train, y_test = train_test_split(X, Y, test_size = val_size)
	#Using logistic regression
	clf = LogisticRegression()
	clf.fit(X_train, y_train)
	
	
	
	
	
	
	