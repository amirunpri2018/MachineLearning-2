#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

#Creating a new figure
plt.figure()

#Initializing x and y values
x_values = np.linspace(-4, 4, 5)
y_values = (0.5 * x_values) + 30

#Plotting the values of x and y in the output
plt.plot(x_values, y_values, '-y', lw = 2)

#Annotating the graph
plt.annotate('y=30+(0.5)x', xy=(0, 30), xytext=(2, 29.5),
            arrowprops = dict(facecolor='black', shrink=0.05))

#Inserting proper labels and title
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Question-1(a)')

#Displaying the graph in the output
plt.show()