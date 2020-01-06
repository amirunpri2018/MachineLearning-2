import math
import numpy as np
import matplotlib.pyplot as plt
from getDataset import getDataSet
from sklearn.linear_model import LogisticRegression


# Starting codes


# step 1: generate dataset that includes both positive and negative samples,
# where each sample is described with two features.
# 250 samples in total.


[X, y] = getDataSet()  # note that y contains only 1s and 0s

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()


def func_DisplayData(dataSamplesX, dataSamplesY, chartNum, titleMessage):
    idx1 = (dataSamplesY == 0).nonzero()  # object indices for the 1st class
    idx2 = (dataSamplesY == 1).nonzero()
    ax = fig.add_subplot(1, 3, chartNum)
    # no more variables are needed
    plt.plot(dataSamplesX[idx1, 0], dataSamplesX[idx1, 1], 'r*')
    plt.plot(dataSamplesX[idx2, 0], dataSamplesX[idx2, 1], 'b*')
    # axis tight
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(titleMessage)


# plotting all samples
func_DisplayData(X, y, 1, 'All samples')

# number of training samples
nTrain = 120


######################PLACEHOLDER 1#start#########################
# choosing random elements from the dataset
maxIndex = len(X)
randomTrainingSamples = np.random.choice(maxIndex, nTrain, replace = False)

# initializing the training and testing lists    
x_train = []
y_train = []
x_test = []
y_test = []

# getting the values for training and testing lists
for i in range(maxIndex):
    if i in randomTrainingSamples:
        x_train.append(X[i])
        y_train.append(y[i])
    else:
        x_test.append(X[i])
        y_test.append(y[i])

# converting the training and testing lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
####################PLACEHOLDER 1#end#########################

# plot the samples you have pickup for training, check to confirm that both negative
# and positive samples are included.
func_DisplayData(x_train, y_train, 2, 'training samples')
func_DisplayData(x_test, y_test, 3, 'testing samples')

# show all charts
plt.show()


# step 2: train logistic regression models


######################PLACEHOLDER2 #start#########################
# using self made model for training
def Sigmoid(x):
    g = float(1.0 / float((1.0 + math.exp(-1.0 * x))))
    return g

def Prediction(theta, X):
    hyp = 0
    for i in range(len(theta)):
        hyp += X[i]*theta[i]
    return Sigmoid(hyp)

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
    sumErrors = 0
    for i in range(m):
        xi = X[i]
        xij = xi[j]
        hi = Prediction(theta,X[i])
        error = (hi - Y[i])*xij
        sumErrors += error
    m = len(X)
    constant = float(alpha)/float(m)
    J = constant * sumErrors
    return J

def Cost_Function(X,Y,theta,m):
    sumOfErrors = 0
    for i in range(m):
        xi = X[i]
        est_yi = Prediction(theta,xi)
        if Y[i] == 1:
            error = Y[i] * math.log(est_yi)
        elif Y[i] == 0:
            error = (1-Y[i]) * math.log(1-est_yi)
        sumOfErrors += error
    const = -1/m
    J = const * sumOfErrors
    return J

def Gradient_Descent(X,Y,theta,m,alpha):
    new_theta = []
    for j in range(len(theta)):
        deltaF = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
        new_theta_value = theta[j] - deltaF
        new_theta.append(new_theta_value)
    return new_theta

# initial model parameters
theta = [0,0,0]
# learning rates
alpha = 0.1
# maximal iterations
max_iteration = 2000

# getting the values of x0 for training dataset
train_xValues = np.ones((len(x_train), 3))
train_xValues[:, 1:3] = x_train[:,:]
train_yValues = y_train

arrCost = []
m = len(train_xValues) # number of samples

for x in range(max_iteration):
    # call the functions for gradient descent method
    new_theta = Gradient_Descent(train_xValues,train_yValues,theta,m,alpha)
    theta = new_theta
    # calculating the cost function
    arrCost.append(Cost_Function(train_xValues,train_yValues,theta,m))
    if x % 200 == 0:
        print("Cost at iteration",x,":",Cost_Function(train_xValues,train_yValues,theta,m))


# using sklearn class for training
logReg = LogisticRegression()
# call the function fit() to train the class instance
logReg.fit(x_train,y_train)
coeffs = logReg.coef_ # coefficients
intercept = logReg.intercept_ # bias
bHat = np.hstack((np.array([intercept]), coeffs)) # model parameters
######################PLACEHOLDER2 #end #########################


# step 3: Use the model to get class labels of testing samples.
 

######################PLACEHOLDER3 #start#########################
# predicting the values using self made model
# appending the values of X0 to the testing dataset
test_xValues = np.ones((len(x_test), 3))
test_xValues[:, 1:3] = x_test[:,:]

# getting the values of the ypred
test_yValues = test_xValues.dot(theta)
for i in range(len(test_yValues)):
    test_yValues[i] = Sigmoid(test_yValues[i])
test_yValues = (test_yValues >= 0.5).astype(int)


# predicting the values using scikit learn library
test_yValues_scikit = test_xValues.dot(np.transpose(bHat))
for i in range(len(test_yValues_scikit)):
    test_yValues_scikit[i] = Sigmoid(test_yValues_scikit[i])
test_yValues_scikit = (test_yValues_scikit >= 0.5).astype(int)
######################PLACEHOLDER 3 #end #########################


# step 4: evaluation


# function for calculating the confusion matrix
def func_calConfusionMatrix(predY, trueY):
    # finding the confusion matrix
    labels = len(np.unique(trueY))
    conf_matr = np.zeros(shape = (labels, labels))
    predY = np.transpose(predY)
    trueY = np.transpose(trueY)
    
    for i in range(len(trueY)):
        for j in range(len(trueY[i])):
            conf_matr[trueY[i][j]][predY[i][j]] += 1  
    
    # finding the accuracy of the model        
    sum_of_diag = 0
    sum_of_elem = 0
    for i in range(len(conf_matr)):
        for j in range(len(conf_matr[i])):
            if i == j:
                sum_of_diag += conf_matr[i][j]
            sum_of_elem += conf_matr[i][j]
    accuracy = sum_of_diag / sum_of_elem
    
    # finding the precision value of the model    
    precision = []        
    for label in range(labels):
        column = conf_matr[:, label]
        precision.append(conf_matr[label, label] / column.sum())
    
    # finding the recall value of the model    
    recall = []
    for label in range(labels):
        row = conf_matr[label, :]
        recall.append(conf_matr[label, label] / row.sum())
        
    return conf_matr, accuracy, precision, recall                                


# evaluating self made model
self_testYDiff = np.abs(test_yValues - y_test)
self_avgErr = np.mean(self_testYDiff)
self_stdErr = np.std(self_testYDiff)
self_score = (len(self_testYDiff) - np.sum(self_testYDiff)) / len(self_testYDiff)

# evaluating scikit model
testYDiff = np.abs(test_yValues_scikit - y_test)
avgErr = np.mean(testYDiff)
stdErr = np.std(testYDiff)
scikit_score = logReg.score(x_test, y_test)

# comparing both the models
print('\nSelf made model average error: {} ({})'.format(self_avgErr, self_stdErr))
print('Scikit learn model average error: {} ({})'.format(avgErr, stdErr))
if self_score > scikit_score:
    print("Self Made Model Wins!")
elif self_score < scikit_score:
    print("Scikit Model Wins!")
else:
    print("Both models perfomed equally well!")
    
# finding the confusion matrices and respective parameters of both the models
print("\nSelf Made Model:")
self_cm, self_acc, self_pre, self_rec = func_calConfusionMatrix(test_yValues, np.array(y_test, dtype = int))
print("Confusion Matrix:\n {} \nAccuracy = {} \nPrecision = {} \nRecall = {}".format(self_cm, self_acc, self_pre, self_rec))

print("\nScikit Model:")
scikit_cm, scikit_acc, scikit_pre, scikit_rec = func_calConfusionMatrix(test_yValues_scikit, np.array(y_test, dtype = int))
print("Confusion Matrix:\n {} \nAccuracy = {} \nPrecision = {} \nRecall = {}".format(scikit_cm, scikit_acc, scikit_pre, scikit_rec))