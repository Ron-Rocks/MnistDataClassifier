import tensorflow as tf 
import tensorflow.keras as keras
from keras.layers import Dense,Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist
(xTrain,yTrain),(xTest,yTest) = mnist.load_data()

xTrain = xTrain/255
xTest = xTest/255

print(xTrain.shape)
model = Sequential()

model.add(Flatten())
model.add(Dense(64,activation = "relu"))
model.add(Dense(128,activation = "relu"))
model.add(Dense(64,activation = "relu"))
model.add(Dense(10,activation = "softmax"))

model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam")

model.fit(x = xTrain,y = yTrain,validation_data = (xTest,yTest),epochs = 3)

prediction = model.predict(xTest)
print(np.argmax(prediction[7]))
plt.imshow(xTest[7])
plt.show()