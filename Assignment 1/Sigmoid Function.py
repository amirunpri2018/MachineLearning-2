#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

#Creating a new figure
plt.figure()

#Initializing x and y values
x_values = np.arange(-10, 10, 0.01)
y_values = 1 / (1 + np.exp(-x_values))

#Plotting the values of x and y in the output
plt.plot(x_values, y_values, '-g', label = 'y=-log(x)', lw = 2)

#Annotating the graph
plt.annotate('y=1/(1+e^(-x))', xy=(0, 0.5), xytext=(2.5, 0.4),
            arrowprops = dict(facecolor='black', shrink=0.05))

#Inserting proper labels and title
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Question-1(d)')

#Displaying the graph in the output
plt.show()