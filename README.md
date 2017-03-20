# creditcard_fraud
Detecting credit card fraud detection. Selecting an optimum threshold with analysis of confusion matrix and ROC curce


This is a dataset from Kaggle. As a typical fraud detection case, the data set is highly unballanced with a ratio of almost 600:1 for negative to positive cases. 

Features for the data are confidential, and are thus just numerical features. They were calculated by applying PCA on the original feature set. 

A typical approach to attack an imbalanced dataset is to - undersample and ajust the threshold based on a ROC analysis. 

1. Undersample the dataset to train a logistic regression classifier on a dataset with 1:1 ratio of positive and negative labels.
2. Use the classifier on the entire dataset
3. Evaluate Recall, and draw ROC curves for different algorithms. 
3. Optimize the Recall-Precision tradeoff as per your requirements. 

Results: 0.87 recall on the complete dataset. 
