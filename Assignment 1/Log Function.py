#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

#Creating a new figure
plt.figure()

#Initializing x and y values
x_values1 = np.arange(0.01, 10, 0.01)
y_values1 = -(np.log(x_values1))
x_values2 = np.arange(-10, 1, 0.01)
y_values2 = -(np.log(1 - x_values2))

#Plotting the values of x and y in the output
plt.plot(x_values1, y_values1, '-r', label = 'y=-log(x)', lw = 2)
plt.plot(x_values2, y_values2, '-b', label = 'y=-log(1-x)', lw = 2)

#Inserting proper labels, title and legend
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Question-1(c)')
plt.legend()

#Displaying the graph in the output
plt.show()