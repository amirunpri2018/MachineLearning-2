from download_data import download_data
import numpy as np
import matplotlib.pyplot as plt
from GD import gradientDescent
# Starting codes for ha2 of CS596

################PLACEHOLDER1 #start##########################
# Test multiple learning rates and report their convergence curves 
ALPHA = 0.01
MAX_ITER = 500
################PLACEHOLDER1 #end############################


# Step-1: Load data and divide it into two subsets, used for training and testing
# Three columns: MATH SAT, VERB SAT, UNI. GPA  
# Convert frame to matrix
sat = download_data('sat.csv', [1, 2, 4]).values 

################PLACEHOLDER2 #start##########################
# Normalize data
min_max_arr = []
for i in range(sat.shape[1]):
    column_values = [row[i] for row in sat]
    min_value = min(column_values)
    max_value = max(column_values)
    min_max_arr.append([min_value, max_value])

for i in range(sat.shape[0]):
    for j in range(sat.shape[1]):
        sat[i][j] = (sat[i][j] - min_max_arr[j][0]) / (min_max_arr[j][1] - min_max_arr[j][0])
################PLACEHOLDER2 #end##########################

# Training data;
satTrain = sat[0:60, :]
# Testing data; 
satTest = sat[60:len(sat),:]


# Step-2: Train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
theta = np.zeros(3)

xValues = np.ones((60, 3))
xValues[:, 1:3] = satTrain[:, 0:2]
yValues = satTrain[:, 2]

# Call the GD algorithm, placeholders in the function gradientDescent()
[theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)

#Visualize the convergence curve
plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()


# Step-3: Testing
testXValues = np.ones((len(satTest), 3)) 
testXValues[:, 1:3] = satTest[:, 0:2]
tVal =  testXValues.dot(theta)


# Step-4: Evaluation
# Calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))