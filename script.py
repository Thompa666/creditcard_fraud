import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.linear_model import LogisticRegression
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
	
	prob_weights = []
	for i in xrange(1, 100):
		data_sample = new_sample(data)
	
		#Separate input data
		Y = data_sample['Class'].values
		data_sample = data_sample.drop('Class', axis = 1)
		X = data_sample.values
	
		#Scale input data
		X = preprocessing.scale(X)
		#Select val size
		val_size = 0.1
		random_st = 7
	
		#Split Train and validation sets
		X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = val_size,
														  random_state = random_st)
		#Using logistic regression
		clf = LogisticRegression()
		clf.fit(X_train, y_train)
		print "Accuracy on Train set %d" %i, round(clf.score(X_train, y_train), 2)
		print "Accuracy on Val set %d" %i, round(clf.score(X_val, y_val), 2)
		prob_tmp = clf.predict_proba(X_train)
		prob_weights.append(prob_tmp)
		
			
	p_0 = 0
	p_1 = 0
	p_final = []
	#print len(prob_weights)
	iter = len(prob_weights[0])
	for j in xrange(0, iter):
		for prediction in xrange(0, len(prob_weights)):
			tmp_0 = prob_weights[prediction][j][0]
			p_0 = tmp_0 + p_0
			tmp_1 = prob_weights[prediction][j][1]
			p_1 = tmp_1 + p_1
			
			if (prediction == (len(prob_weights) - 1)):
				pf_0 = p_0/len(prob_weights)
				pf_1 = p_1/len(prob_weights)
				p_final.append([pf_0, pf_1])
				
				p_0 = 0
				p_1 = 0
	
	print p_final
	
	
	
	
	
	