# APS-Failure-at-Scania-Trucks

The dataset files used for this problem were taken from https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks#. Unfortunately, the training set file was too large (>25MB) to fit in this repository, so the training and testing csv files used for this problem were not included. If you wish to get the files, just download them directly from the given link to your local machine. The classification analysis using tree based methods can be found in the APS_Failure_Analysis.ipynb jupyter notebook. 

# Logistic Model Trees in Weka

I exported the imputed data in the pandas dataframe to .arff file format to use the Logistic Model Trees Classifier in WEKA.
As stated above, these files were too large (>25MB) to include in the repository, so the files named imbalanced_train.arff and imbalanced_test.arff used in the python codes will not be found in this repository. The codes I used to run WEKA with the python wrapper is named WEKA_LMT_Imbalanced.py and WEKA_LMT_SMOTE.py corresponding to building a model on the imbalanced dataset and the balanced dataset respectively. Along with the code there is two html files, which are printouts of the console in Spyder IDE to show the ROC, AUC, Confusion matrix and Misclassification error for the training and test sets. The output is shown in the file named LMT_Imbalanced.pdf and LMT_SMOTE.pdf.
