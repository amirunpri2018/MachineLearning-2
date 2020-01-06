import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scalar

def gradientDescent(X, y, theta, alpha, numIterations):
    '''
    # This function returns a tuple (theta, Cost array)
    '''
    m = len(y)
    arrCost =[];
    transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
    
    for iteration in range(0, numIterations):
        ################PLACEHOLDER3 #start##########################
        # Write your codes to update theta, i.e., the parameters to estimate.
        # Replace the following variables if needed
        hyp = np.dot(theta, transposedX)
        residualError = np.subtract(y, hyp)
        gradient = (1 / m) * np.dot(transposedX, np.subtract(hyp, y))
        change = [alpha * x for x in gradient]
        theta = np.subtract(theta, change)  # or theta = theta - alpha * gradient
        ################PLACEHOLDER3 #end##########################

        ################PLACEHOLDER4 #start##########################
        # Calculate the current cost with the new theta; 
        atmp = (1 / m) * np.sum(residualError ** 2)
        arrCost.append(atmp)
        ################PLACEHOLDER4 #start##########################
        
    return theta, arrCost