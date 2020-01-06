#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

#Creating a new figure
plt.figure()

#Initializing x and y values
x_values = np.arange(-200, 250, 0.1)
y_values = (x_values - 25)**2 + 20

#Plotting the values of x and y in the output
plt.plot(x_values, y_values, '-m', lw = 2)

#Annotating the graph
plt.annotate('y=(x-25)^2 + 20', xy=(200, 30645), xytext=(-50, 40000),
            arrowprops = dict(facecolor='black', shrink=0.05))
plt.annotate('Vertex', xy=(25, 20), xytext=(25, 10000),
            arrowprops = dict(facecolor='black', shrink=0.05))

#Inserting proper labels and title
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Question-1(b)')

#Displaying the graph in the output
plt.show()