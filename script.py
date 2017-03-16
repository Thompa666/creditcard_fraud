import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

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
	
	
	
	
	
	