# Author: Dhaval Harish Sharma
# Red ID: 824654344
# Assignment 5

import numpy as np
import download_data as dl
import matplotlib.pyplot as plt
import sklearn.svm as svm
from conf_matrix import func_confusion_matrix
import pandas as pd
import warnings
import sys

warnings.filterwarnings("ignore")
## step 1: load data from csv file. 
data = dl.download_data('crab.csv').values
print(data[:5])

n = 200
#split data 
S = np.random.permutation(n)
#100 training samples
Xtr = data[S[:100], :6]
Ytr = data[S[:100], 6:]
# 100 testing samples
X_test = data[S[100:], :6]
Y_test = data[S[100:], 6:].ravel()

## step 2 randomly split Xtr/Ytr into two even subsets: use one for training, another for validation.
#############placeholder 1: training/validation #######################
n2 = len(Xtr)
S2 = np.random.permutation(n2)
 
# subsets for training models
x_train= Xtr[S2[:50],:6]
y_train= Ytr[S2[:50],0]
# subsets for validation
x_validation= Xtr[S2[50:],:6]
y_validation= Ytr[S2[50:],0]
#############placeholder end #######################


## step 3 Model selection over validation set
# consider the parameters C, kernel types (linear, RBF etc.) and kernel
# parameters if applicable. 


# 3.1 Plot the validation errors while using different values of C ( with other hyperparameters fixed) 
#  keeping kernel = "linear"
#############placeholder 2: Figure 1#######################
error_min = sys.maxsize
error_min_for_c = 0
c_range =  np.arange(1,10,0.5)
svm_c_error = []
for c_value in c_range:
    model = svm.SVC(kernel='linear', C=c_value)
    model.fit(X=x_train, y=y_train)
    error = 1. - model.score(x_validation, y_validation)
    if error_min > error:
        error_min = error
        error_min_for_c = c_value
    svm_c_error.append(error)
plt.plot(c_range, svm_c_error)
plt.title('Linear SVM')
plt.xlabel('c values')
plt.ylabel('error')
plt.xticks(c_range)
plt.show()
#############placeholder end #######################


# 3.2 Plot the validation errors while using linear, RBF kernel, or Polynomial kernel ( with other hyperparameters fixed) 
#############placeholder 3: Figure 2#######################
best_c_value = error_min_for_c
error_min_for_kernal = 0
kernel_error_min  = sys.maxsize
# print(best_c_value)
kernel_error=[]
kernel_types = ['linear', 'poly', 'rbf']
svm_kernel_error = []
for kernel_value in kernel_types:
    model = svm.SVC(kernel=kernel_value, C=best_c_value)
    model.fit(X=x_train, y=y_train)
    error = 1. - model.score(x_validation, y_validation)
    if kernel_error_min > error:
        kernel_error_min = error
        error_min_for_kernal = kernel_value
    svm_kernel_error.append(error)

plt.plot(kernel_types, svm_kernel_error)
plt.title('SVM by Kernels')
plt.xlabel('Kernel')
plt.ylabel('error')
plt.xticks(kernel_types)
plt.show()
#############placeholder end #######################

print("BEST c value:",error_min_for_c)
print("BEST kernel value:",error_min_for_kernal)

## step 4 Select the best model and apply it over the testing subset 
#############placeholder 4:testing  #######################

best_kernel = error_min_for_kernal
best_c = error_min_for_c # poly had many that were the "best"
model = svm.SVC(kernel=best_kernel, C=best_c)
model.fit(X=x_train, y=y_train)
#############placeholder end #######################


## step 5 evaluate your results in terms of accuracy, real, or precision. 

#############placeholder 5: metrics #######################
# func_confusion_matrix is not included
# You might re-use this function for the Part I. 
y_pred = model.predict(X_test)
conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(Y_test, y_pred)
# print(Y_test)
# print(y_pred)
print("Confusion Matrix: ")
print(conf_matrix)
print("Average Accuracy: {}".format(accuracy))
print("Per-Class Precision: {}".format(precision_array))
print("Per-Class Recall: {}".format(recall_array))
#############placeholder end #######################

#############placeholder 6: success and failure examples #######################
# Success samples: samples for which you model can correctly predict their labels
# Failure samples: samples for which you model can not correctly predict their labels
correct_preds = []
wrong_preds = []
for i in range(0,len(Y_test)):
    temp=[]
    temp.append(X_test[i])
    temp.append(Y_test[i])
    temp.append(y_pred[i])
    if Y_test[i] != y_pred[i] and len(wrong_preds)<5 :
        wrong_preds.append(temp)
    if Y_test[i] == y_pred[i] and len(correct_preds)<5:
        correct_preds.append(temp)
frame1 = pd.DataFrame(correct_preds,columns=["Crab_features","Ground Truth","Prediction"])
frame2 = pd.DataFrame(wrong_preds,columns=["Crab_features","Ground Truth","Prediction"])
print("--------Success Examples--------")
print(frame1)
print("--------Failure Examples--------")
print(frame2)
#############placeholder end #######################