import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np 

mnist = tf.keras.datasets.mnist
(xTrain,yTrain),(xTest,yTest) = mnist.load_data()

xTrain = (xTrain.reshape(-1,784))/255
xTest = (xTest.reshape(-1,784))/255

inputLayer = keras.Input(shape = (784,))

x = layers.Dense(64,activation = "relu")(inputLayer)

x =layers.Dense(128,activation = "relu")(x)

x = layers.Dense(64,activation = "relu")(x)

outputLayer = layers.Dense(10,activation = "softmax")(x)

model = keras.Model(inputs = inputLayer,outputs = outputLayer)


model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam",metrics = ["accuracy"])

model.fit(xTrain,yTrain,validation_split = 0.3,epochs = 5)

prediction = model.predict(xTest)

print(np.argmax(prediction[9]))
plt.imshow(xTest[9].reshape(28,28))
plt.show()
