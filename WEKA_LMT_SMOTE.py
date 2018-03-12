# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:39:57 2018

@author: Andrew McLean
Description: Training and Testing a Logistic Model Tree that compensates for 
class imbalancing using SMOTE by using the WEKA python wrapper.
"""

import weka.core.jvm as jvm
import weka.core.converters as converters
import weka.plot.classifiers as plcls
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random

# Starts the Java Handler
jvm.start()

# Loads the Data 
train = converters.load_any_file("imbalanced_train.arff")
test = converters.load_any_file("imbalanced_test.arff")

train.class_is_last()
test.class_is_last()

# Setting the number of iterations performed by Logit Boost
cls = Classifier(classname="weka.classifiers.trees.LMT", options=["-B", "-I", "10"])

# 5 Fold Cross Validation Error
evl = Evaluation(train)
evl.crossvalidate_model(cls, train, 5, Random(1))

# Prints Out Confusion Matrix along with other summary statistics
print("LMT (imbalanced classes) CV = 5 Error: %.2f%%" % (evl.percent_incorrect))
print(evl.matrix()) #Confusion Matrix

# Plots ROC
plcls.plot_roc(evl, class_index=[0, 1], wait=True)

# Extra Summary
print(evl.summary())
print(evl.class_details())

# Evaluate the classifier on test set
cls.build_classifier(train)
tevl = Evaluation(test)
tevl.test_model(cls, test)

# Prints Out Confusion Matrix along with other summary statistics
print("LMT (imbalanced classes) Test Error: %.2f%%" % (tevl.percent_incorrect))
print(tevl.matrix()) #Confusion Matrix

# Plots ROC 
plcls.plot_roc(tevl, class_index=[0, 1], wait=True)

# Extra Summary
print(tevl.summary())
print(tevl.class_details())

# Stops the Java Handler
jvm.stop()