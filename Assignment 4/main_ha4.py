# Author: Dhaval Harish Sharma
# Red ID: 824654344
# Assignment 4

import numpy as np
import matplotlib.pyplot as plt
from util import func_confusion_matrix
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop 

# load (downloaded if needed) the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# transform each image from 28 by28 to a 784 pixel vector
pixel_count = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')

# normalize inputs from gray scale of 0-255 to values between 0-1
x_train = x_train / 255
x_test = x_test / 255

# Please write your own codes in responses to the homework assignment 4
validation_data = x_train[50000:], y_train[50000:] 
x_train, y_train = x_train[:50000], y_train[:50000]

model1 = tf.keras.models.Sequential(
        [ tf.keras.layers.Dense(512,activation='relu',input_dim=784),
         tf.keras.layers.Dense(10,activation='softmax') ])
model2 = tf.keras.models.Sequential(
        [ tf.keras.layers.Dense(512,activation='relu',input_dim=784),
         tf.keras.layers.Dense(64,activation='relu'),
         tf.keras.layers.Dense(10,activation='sigmoid') ])
model3 = tf.keras.models.Sequential(
        [ tf.keras.layers.Dense(128,activation='relu',input_dim=784),
         tf.keras.layers.Dense(10,activation='linear') ])

model1.summary()
model2.summary()
model3.summary()


model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model2.compile(optimizer=RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model3.compile(optimizer=RMSprop(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model1.fit(x_train,y_train,epochs=5,validation_data=validation_data)
model2.fit(x_train,y_train,epochs=5,validation_data=validation_data)
model3.fit(x_train,y_train,epochs=5,validation_data=validation_data) 

y_pred2 = model2.predict(x_test)


yli_pred2 = []
for i in range(len(y_test)):
    yli_pred2.append(list(y_pred2[i]).index(np.max(y_pred2[i])))


result = func_confusion_matrix(y_test,yli_pred2)
conf_matrix = result[0]
accuracy = result[1]
recall = result[2]
precision = result[3]

print("Confusion Matrix : \n",conf_matrix)
print("\n\nAccuracy : ",accuracy)
print("\n\n")
for i in range(10):
    print("Class ",i," Recall : ",recall[i])
print("\n\n")
for i in range(10):
    print("Class ",i," Precision : ",precision[i])


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

good = []
bad = []


print("\n\nClassified Images")
print("True Label\tPredicted Label")
i=0
j=0
while(i<10):
    if y_test[j]==yli_pred2[j]:
        print(y_test[j],"\t\t",yli_pred2[j])
        good.append(j)
        i += 1
    j += 1
    
fig1=plt.figure(figsize=(10, 4))
columns = 5
rows = 2
for i in range(1, columns*rows +1):
    fig1.add_subplot(rows, columns, i)
    plt.imshow(x_test[good[i-1]])
plt.show()  


print("\n\nMisclassified Images")
print("True Label\tPredicted Label")
i=0
j=0
while(i<10):
    if y_test[j]!=yli_pred2[j]:
        print(y_test[j],"\t\t",yli_pred2[j])
        bad.append(j)
        i += 1
    j = j + 1

fig2=plt.figure(figsize=(10, 4))
columns = 5
rows = 2   
for i in range(1, columns*rows +1):
    fig2.add_subplot(rows, columns, i)
    plt.imshow(x_test[bad[i-1]])
plt.show()