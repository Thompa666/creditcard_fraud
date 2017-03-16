import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, recall_score

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

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[10]) 
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print


if __name__ == '__main__':
	file = 'creditcard.csv'
	data = pd.read_csv(file, delimiter=',')
	
	#Uncomment to see simple graphs of the given data	
	#simple_graphing(data)		
	
	recall_matrix = []
	for i in xrange(1, 100):
		data_sample = new_sample(data)
	
		#Separate input data
		Y = data_sample['Class'].values
		data_sample = data_sample.drop('Class', axis = 1)
		X = data_sample.values
	
		#Scale input data
		#X = preprocessing.scale(X)
		#Select val size
		val_size = 0.3
		random_st = 7
	
		#Split Train and validation sets
		X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = val_size,
														  random_state = random_st)
		#Using logistic regression
		clf = LogisticRegression(C = 0.1)
		clf.fit(X_train, y_train)
		y_train_pred = clf.predict(X_train)
		y_val_pred = clf.predict(X_val)
		y = clf.predict(X)
		print "Recall on Train set %d" %i, round(recall_score(y_train, y_train_pred), 2)
		print "Recall on Val set %d" %i, round(recall_score(y_val, y_val_pred), 2)
		recall_tmp = round(recall_score(Y, y), 2)
		print "Recall on X %d" %i, recall_tmp
		recall_matrix.append(recall_tmp)
		
	avg_recall = sum(recall_matrix)/len(recall_matrix)
	print "Average recall on undersampled set:", avg_recall
	
	conf = confusion_matrix(Y, y)
	print "Confusion matrix for undersampled set (Last iteration):"
	print conf
	
	print "Predicitng on the complete set using the model" 
	Y = data['Class'].values
	data= data.drop('Class', axis = 1)
	X = data.values
	
	#Scale input data
	#X = preprocessing.scale(X)
	#Select val size
	val_size = 0.3
	random_st = 7
	
	#Split Train and validation sets
	X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = val_size,
													  random_state = random_st)
		
	y_train_pred = clf.predict(X_train)
	y_val_pred = clf.predict(X_val)
	y = clf.predict(X)
	
	print "Recall on Complete Train set" , round(recall_score(y_train, y_train_pred), 2)
	print "Recall on Complete Val set" , round(recall_score(y_val, y_val_pred), 2)
	print "Recall on Complete X", round(recall_score(Y, y), 2)
	
	conf_complete = confusion_matrix(Y, y)
	print "Confusion matrix for undersampled set (Last iteration):"
	#print conf_complete
	
	labels = ['Legit', 'Fraud']
	print_cm(conf_complete, labels)
	
	#p_0 = 0
	#p_1 = 0
	#p_final = []
	
	#iter = len(prob_weights[0])
	#for j in xrange(0, iter):
	#	for prediction in xrange(0, len(prob_weights)):
	#		tmp_0 = prob_weights[prediction][j][0]
	#		p_0 = tmp_0 + p_0
	#		tmp_1 = prob_weights[prediction][j][1]
	#		p_1 = tmp_1 + p_1
	#		
	#		if (prediction == (len(prob_weights) - 1)):
	#			pf_0 = p_0/len(prob_weights)
	#			pf_1 = p_1/len(prob_weights)
	#			p_final.append([pf_0, pf_1])
				
	#			p_0 = 0
	#			p_1 = 0
	
	#class_0, class_1 = [], []
	#for prob in p_final:
	#	class_0.append(prob[0])
	#	class_1.append(prob[1])
		
	#p_final = pd.DataFrame({'class_0': class_0, 'class_1': class_1})
	#print 'Aggregated final prediction matrix:'
	#thrs = 0.5
	#print 'Threshold:', thrs 	
	#p_final['prediction'] = np.where((p_final['class_1'] > thrs), 1, 0)
	#print p_final
	

	
	
	
	
	
	
	
	
	