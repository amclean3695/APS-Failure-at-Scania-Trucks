# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:39:57 2018

@author: Andrew McLean
Description: Training and Testing a Logistic Model Tree that compensates for 
class imbalancing using SMOTE by using the WEKA python wrapper.
"""

import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.classifiers import Classifier, FilteredClassifier, Evaluation
from weka.core.classes import Random
from weka.filters import Filter
import weka.plot.classifiers as plcls
import weka.core.packages as packages


# Starts the Java Handler with packages set to True
jvm.start(packages = True)

packages.install_package("SMOTE")

# Loads the Data 
train = converters.load_any_file("imbalanced_train.arff")
test = converters.load_any_file("imbalanced_test.arff")

train.class_is_last()
test.class_is_last()

# Minority Class is getting Sampled 5x
smote = Filter(classname="weka.filters.supervised.instance.SMOTE", options = ["-P", "500.0"])

# Base Classifier
cls = Classifier(classname="weka.classifiers.trees.LMT", options=["-B", "-I", "10"])

# Filtered Classifier
fc = FilteredClassifier()
fc.filter = smote
fc.classifier = cls

# 5 Fold K cross validation
evl = Evaluation(train)
evl.crossvalidate_model(fc, train, 5, Random(1))

# Prints Out Confusion Matrix along with other summary statistics
print("LMT (SMOTE balanced classes) CV = 5 Error: %.2f%%" % (evl.percent_incorrect))
print(evl.matrix()) #Confusion Matrix

# Plots ROC
plcls.plot_roc(evl, class_index=[0, 1], wait=True)

# Extra Summary
print(evl.summary())
print(evl.class_details())

# Evaluate the classifier on test set
fc.build_classifier(train)
tevl = Evaluation(test)
tevl.test_model(fc, test)

# Prints Out Confusion Matrix along with other summary statistics
print("LMT (SMOTE balanced classes) Test Error: %.2f%%" % (tevl.percent_incorrect))
print(tevl.matrix()) #Confusion Matrix

# Plots ROC 
plcls.plot_roc(tevl, class_index=[0, 1], wait=True)

# Extra Summary
print(tevl.summary())
print(tevl.class_details())

# Stops the Java Handler
jvm.stop()